from typing import TypedDict, List, Optional, Sequence, Tuple, Dict, Any
import numpy as np
import trimesh
import time
import logging
from scipy import sparse
from scipy.spatial import cKDTree
from genies.meshutils.geometry.mesh_data.mesh_relationship import TmMeshRelationship, BarycentricMapping
from genies.meshutils.geometry.voxelize import tm_build_solid_field
from genies.meshutils.geometry.smooth import tm_delta_mush

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
# Dictionary keys are camelCase on purpose: these payloads are dumped verbatim
# to the JSON sidecar and consumed as-is downstream (UsdWearable.get_anchor_mappings
# -> GUPM container -> RuntimeMeshAnchorModel), so the keys mirror the
# RuntimeMeshAnchor.fbs field names exactly, end to end.

class PointAnchor(TypedDict):
    """Defines the barycentric connection for a single vertex.

    Matches the PointAnchor struct in the RuntimeMeshAnchor flatbuffer schema:
    a surface triangle, barycentric coordinates on it, and a blend weight.
    """
    faceId: int
    baryCoords: List[float]
    weight: float


class PointAnchorSet(TypedDict):
    """Groups point anchors into contiguous memory blocks.

    Matches the PointAnchorSet table: a flat list of anchors, per-point offsets
    into it (CSR layout), and the bind-pose gap vector of every point.
    """
    anchors: List[PointAnchor]
    anchorOffsets: List[int]
    gapVectors: List[List[float]]


class AnchorFace(TypedDict):
    """Stores the bind-pose triangle of one anchor surface face.

    Raw bind points rather than a pre-inverted basis: a basis cannot be
    converted across a handedness change, so the runtime rebuilds and inverts
    the bases from the converted points once at load. sourceIndex references the
    anchorSurfaceSources entry the face belongs to.
    """
    sourceIndex: int
    sourceFaceId: int
    point0: List[float]
    point1: List[float]
    point2: List[float]


class AnchorSurfaceSource(TypedDict):
    """Identifies one template mesh contributing faces to the anchor surface.

    The runtime resolves the surface geometry by meshId and validates it by
    faceCount.
    """
    meshId: str
    faceCount: int


class AnchorSurfacePatch(TypedDict):
    """Replacement positions for a subset of an anchor surface source.

    The bulk a wearable leaves on the template. The runtime deforms the patch
    like any other anchored point set, then writes the result into its source
    before refitting whatever is layered on top, in equipping order.
    """
    positions: List[List[float]]
    sparseIndices: List[int]
    vertexAnchors: PointAnchorSet


class SmoothingGraph(TypedDict):
    """Static neighbor connectivity for the runtime delta smoothing pass.

    Mesh edges plus bind-pose proximity links, in the same flat CSR layout as
    PointAnchorSet.
    """
    neighborIndices: List[int]
    neighborOffsets: List[int]


class RuntimeMeshAnchor(TypedDict):
    """Contains the complete deformation payload of one wearable mesh."""
    vertexAnchors: PointAnchorSet
    jointAnchors: PointAnchorSet
    anchorSurface: List[AnchorFace]
    anchorSurfaceSources: List[AnchorSurfaceSource]
    smoothingGraph: SmoothingGraph
    layerStiffness: float


class AnchorLayer(TypedDict, total=False):
    """Another wearable equipped alongside this one, and how hard it holds its shape.

    Not a payload table: nothing here is baked. layerStiffness (0 to 1) is a
    choice, not a material property -- a boot is usually high and a legging
    low, but the same asset can want another value in another combination.
    Omitted, the payload's authored value is used; given, it overrides it for
    this combination only. What a pair does with two of these is
    layer_press_share.
    """
    anchorMapping: RuntimeMeshAnchor
    layerStiffness: float


# =============================================================================
# BAKE AND SAVE FUNCTIONS
# =============================================================================

def _pack_anchor_set(
    target_positions: np.ndarray,
    mapping_data: BarycentricMapping,
    refit_mesh: trimesh.Trimesh,
    max_influences: int
) -> PointAnchorSet:
    """Formats raw barycentric mapping data into the payload layout.

    Vectorized gap calculation, then a plain loop for the JSON formatting.
    Assumes the weights are pre-normalized by the spatial solver.

    Args:
        target_positions: Array of local space positions to anchor.
        mapping_data: Raw barycentric relationship results.
        refit_mesh: The base mesh geometry.
        max_influences: Maximum allowed anchors per point.

    Returns:
        PointAnchorSet: Structured anchor payload.
    """
    # 1. Extract and truncate the mapping arrays to the influence limit
    faces = mapping_data.face_indices[:, :max_influences]
    barys = mapping_data.barycentric_coords[:, :max_influences, :]
    weights = mapping_data.influence_weights[:, :max_influences]

    # 2. Vectorized gap calculation: blended surface point per target, then the
    # offset from it. barys (N, K, 3), tri_verts (N, K, 3, 3) -> (N, K, 3) -> (N, 3)
    tri_verts = refit_mesh.vertices[refit_mesh.faces[np.maximum(faces, 0)]]
    surface_pts = np.einsum('nkv,nkvd->nkd', barys, tri_verts)
    anchor_pos = np.einsum('nkd,nk->nd', surface_pts, weights)
    gaps = target_positions - anchor_pos

    # 3. Serialization: one flat anchor list, offsets delimiting each point
    anchors = []
    anchor_offsets = [0]
    for i in range(len(target_positions)):
        for k in range(faces.shape[1]):
            if faces[i, k] != -1 and weights[i, k] > 0:
                anchors.append({
                    "faceId": int(faces[i, k]),
                    "baryCoords": [float(b) for b in barys[i, k]],
                    "weight": float(weights[i, k])
                })
        anchor_offsets.append(len(anchors))

    return {"anchors": anchors, "anchorOffsets": anchor_offsets, "gapVectors": gaps.tolist()}


def _shrink_gaps_to_body(
    anchor_set: PointAnchorSet,
    refit_mesh: trimesh.Trimesh,
    gap_size: float
) -> None:
    """Shortens every gap vector by the clearance measured at its landing point.

    A length per landing rather than a scale, so a panel keeps its modelled
    thickness: both its faces land on the same body ring and lose the same
    amount. The clearance is the smallest gap landing on that ring -- a
    face-to-vertex-to-face hop on the body, with no radius to pick.

    Args:
        anchor_set: The packed anchors. gapVectors is modified in place.
        refit_mesh: The base mesh geometry, faceIds still indexing its face list.
        gap_size: Fraction of the clearance kept. 1.0 leaves the gaps as authored.
    """
    anchors = anchor_set["anchors"]
    gaps = np.asarray(anchor_set["gapVectors"], dtype=np.float64)
    if not anchors or len(gaps) == 0:
        return

    # 1. Unpack the anchors and find the points that have any
    face_ids = np.fromiter((a["faceId"] for a in anchors), dtype=np.int64)
    weights = np.fromiter((a["weight"] for a in anchors), dtype=np.float64)
    offsets = np.asarray(anchor_set["anchorOffsets"], dtype=np.int64)
    lengths = np.linalg.norm(gaps, axis=1)
    mapped = np.flatnonzero(np.diff(offsets) > 0)
    if len(mapped) == 0:
        return

    # 2. Landing face of each point: its strongest influence
    landing = np.fromiter(
        (int(face_ids[offsets[i] + int(np.argmax(weights[offsets[i]:offsets[i + 1]]))]) for i in mapped),
        dtype=np.int64, count=len(mapped))
    corners = refit_mesh.faces[landing]

    # 3. Smallest gap landing on each body vertex, read back per landing face
    per_body_vertex = np.full(len(refit_mesh.vertices), np.inf)
    np.minimum.at(per_body_vertex, corners.ravel(), np.repeat(lengths[mapped], 3))
    clearance = per_body_vertex[corners].min(axis=1)

    # 4. Shorten every gap by the share of that clearance being removed
    kept = lengths[mapped] - (1.0 - gap_size) * clearance
    gaps[mapped] *= (np.maximum(kept, 0.0) / np.maximum(lengths[mapped], 1e-12))[:, None]
    anchor_set["gapVectors"] = gaps.tolist()

    logger.info(f"Shrunk gaps to gap_size {gap_size:.3f}: clearance median {float(np.median(clearance)):.5f} "
                f"against gap median {float(np.median(lengths[mapped])):.5f}.")


def _pack_smoothing_graph(wearable_mesh: trimesh.Trimesh, proximity_ratio: float) -> SmoothingGraph:
    """Builds and packs the static smoothing graph into flat CSR arrays.

    The wearable's own edges plus proximity links between bind-pose vertices
    closer than proximity_ratio times the mean edge, which weld doubled seams
    and tie overlapping panels together. Bind-only, so it is baked once and the
    runtime performs no spatial queries.
    """
    num_verts = len(wearable_mesh.vertices)
    edges = np.asarray(wearable_mesh.edges_unique, dtype=np.int64)

    # 1. Proximity links (seams, overlaps)
    if proximity_ratio > 0.0:
        radius = float(wearable_mesh.edges_unique_length.mean()) * proximity_ratio
        pairs = cKDTree(wearable_mesh.vertices).query_pairs(r=radius, output_type='ndarray')
        if len(pairs):
            edges = np.vstack([edges, pairs.astype(np.int64)])

    # 2. Symmetrize and deduplicate, so a short edge is not counted twice
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    keys = np.unique(rows * num_verts + cols)
    rows, cols = keys // num_verts, keys % num_verts

    # 3. CSR offsets from the per-vertex counts (unique keys arrive sorted by vertex)
    counts = np.bincount(rows, minlength=num_verts)
    return {"neighborIndices": cols.tolist(),
            "neighborOffsets": np.concatenate([[0], np.cumsum(counts)]).tolist()}


def bake_anchor_mapping(
    refit_mesh: trimesh.Trimesh,
    wearable_mesh: trimesh.Trimesh,
    anchor_surface_sources: Optional[List[AnchorSurfaceSource]] = None,
    joint_bind_positions: Optional[np.ndarray] = None,
    normal_threshold: float = -1.0,
    max_search_dist: float = 0.0,
    max_influences: int = 4,
    bi_directional_search: bool = False,
    proximity_ratio: float = 1.0,
    gap_size: float = 1.0,
    layer_stiffness: float = 0.0,
    bake_patch: bool = True,
    patch_params: Optional[Dict[str, Any]] = None
) -> RuntimeMeshAnchor:
    """Computes and packages the anchor relationships of a wearable to the template.

    Projects the wearable (and joints) onto the template, packs the anchors and
    gaps, bakes the patch the wearable leaves on the template, then trims the
    anchor surface to the referenced faces.

    Args:
        refit_mesh: The template at bind pose.
        wearable_mesh: The wearable at bind pose, in the same space.
        anchor_surface_sources: One {"meshId", "faceCount"} per template mesh;
            the faceCounts must sum to the template's face count.
        joint_bind_positions: Optional joints to anchor.
        normal_threshold: Normal correspondence threshold of the projection (-1 bypasses it).
        max_search_dist: Distance limit of the projection.
        max_influences: Maximum allowed anchors per point.
        bi_directional_search: Search from both meshes for the best correspondence.
        proximity_ratio: Proximity link radius of the smoothing graph, in mean edges.
        gap_size: Share of the authored clearance kept, 1.0 leaves the gaps as authored.
        layer_stiffness: Authored default of how much the wearable keeps its shape
            under another one, 0 to 1. Carried in the payload, not used by the bake.
        bake_patch: Also bake the bulk this wearable leaves on the template, on the
            first source as "patch". Without it the wearable cannot take part in
            layering at all.
        patch_params: Overrides forwarded to bake_anchor_patch.

    Returns:
        RuntimeMeshAnchor: Structured mapping payload.
    """
    start_time = time.time()
    anchor_surface_sources = anchor_surface_sources or []
    relationship = TmMeshRelationship(refit_mesh)

    # 1. Project the wearable vertices onto the template
    vertex_mapping = relationship.compute_barycentric_mapping_planar(
        tgt_verts=wearable_mesh.vertices, tgt_normals=wearable_mesh.vertex_normals,
        normal_threshold=normal_threshold, max_search_dist=max_search_dist,
        max_influences=max_influences, bi_directional_search=bi_directional_search)

    # 2. Project the joints, with dummy normals since joints have none
    joint_mapping = None
    if joint_bind_positions is not None and len(joint_bind_positions) > 0:
        dummy_normals = np.zeros_like(joint_bind_positions)
        dummy_normals[:, 1] = 1.0
        joint_mapping = relationship.compute_barycentric_mapping_planar(
            tgt_verts=joint_bind_positions, tgt_normals=dummy_normals, normal_threshold=-1.0,
            max_search_dist=max_search_dist, max_influences=max_influences, bi_directional_search=False)

    # 3. The anchor surface: every template face as raw bind points, tagged with its source
    all_tri_verts = refit_mesh.vertices[refit_mesh.faces]
    if not anchor_surface_sources:
        anchor_surface_sources = [{"meshId": "", "faceCount": int(len(refit_mesh.faces))}]

    face_counts = np.array([int(s["faceCount"]) for s in anchor_surface_sources], dtype=np.int64)
    if face_counts.sum() != len(all_tri_verts):
        raise ValueError(f"anchor_surface_sources faceCount sum ({int(face_counts.sum())}) does not "
                         f"match refit mesh face count ({len(all_tri_verts)}).")
    source_indices = np.searchsorted(np.cumsum(face_counts), np.arange(len(all_tri_verts)), side='right')
    source_starts = np.concatenate(([0], np.cumsum(face_counts)[:-1]))

    anchor_surface = [{
        "sourceIndex": int(source_indices[i]),
        "sourceFaceId": int(i - source_starts[source_indices[i]]),
        "point0": verts[0].tolist(), "point1": verts[1].tolist(), "point2": verts[2].tolist()
    } for i, verts in enumerate(all_tri_verts)]

    # 4. Pack the vertex anchors, and tighten the gaps while the faceIds still index the template
    vertex_anchors = _pack_anchor_set(wearable_mesh.vertices, vertex_mapping, refit_mesh, max_influences)
    if gap_size != 1.0:
        _shrink_gaps_to_body(vertex_anchors, refit_mesh, gap_size)

    # 5. Pack the joint anchors
    if joint_mapping is not None:
        joint_anchors = _pack_anchor_set(joint_bind_positions, joint_mapping, refit_mesh, max_influences)
    else:
        joint_anchors = {"anchors": [], "anchorOffsets": [0], "gapVectors": []}

    # 6. Bake the patch, before the trim: its anchors address the full face list too
    if bake_patch:
        sparse_indices, positions = bake_anchor_patch(refit_mesh, wearable_mesh, **(patch_params or {}))
        patch = pack_anchor_patch(refit_mesh, sparse_indices, positions, max_influences)
        anchor_surface_sources = [dict(source) for source in anchor_surface_sources]
        anchor_surface_sources[0]["patch"] = patch

    # 7. Trim the anchor surface to the referenced faces. faceId indexes the
    # trimmed list, sourceFaceId keeps addressing the full source mesh.
    patch_anchors = [a for s in anchor_surface_sources if s.get("patch")
                     for a in s["patch"]["vertexAnchors"]["anchors"]]
    used = sorted({a["faceId"] for a in vertex_anchors["anchors"]} |
                  {a["faceId"] for a in joint_anchors["anchors"]} |
                  {a["faceId"] for a in patch_anchors})
    remap = {old: new for new, old in enumerate(used)}
    anchor_surface = [anchor_surface[i] for i in used]
    for a in vertex_anchors["anchors"] + joint_anchors["anchors"] + patch_anchors:
        a["faceId"] = remap[a["faceId"]]

    # 8. The smoothing graph, so the runtime performs no spatial queries
    smoothing_graph = _pack_smoothing_graph(wearable_mesh, proximity_ratio)
    logger.info(f"Finished bake_anchor_mapping in {time.time() - start_time:.4f} seconds.")

    return {
        "vertexAnchors": vertex_anchors,
        "jointAnchors": joint_anchors,
        "anchorSurface": anchor_surface,
        "anchorSurfaceSources": anchor_surface_sources,
        "smoothingGraph": smoothing_graph,
        "layerStiffness": float(np.clip(layer_stiffness, 0.0, 1.0))
    }


# =============================================================================
# PATCH BAKE
# =============================================================================
# Machinery behind the patch bake, settled on real assets. Every distance is a
# ratio of the template's diagonal, so the same numbers hold whatever the unit
# of the scene and whatever the resolution of the template.
_PATCH_VOXEL_RATIO = 0.002           # Voxel pitch of the containment field.
_PATCH_CLEARANCE_RATIO = 0.002       # How far the patch stands proud of the wearable; also one storey of a stack.
_PATCH_OVERSAMPLE = 5.0              # Wearable surface samples per template vertex area.
_PATCH_CLOSE_RATIO = 0.085           # Morphological closing of the coverage, to claim the grooves a wearable bridges.
_PATCH_DIRECTION_REACH_RATIO = 0.75  # Relaxation of the travel directions: normals converge inside a concavity.
_PATCH_FEATHER_ITERATIONS = 5        # Relaxation passes on the displacements.
_PATCH_MUSH_ITERATIONS = 10          # Delta mush passes over the folded faces.
_PATCH_MUSH_FALLOFF_RATIO = 0.1      # How far the mush reaches past its mask.
_PATCH_MIN_EDGE_RATIO = 0.25         # Shortest a displaced template edge may become, as a fraction of itself:
                                     # converging directions (crotch, armpit) otherwise cross and fold. 0 disables.


def _patch_heights_from_wearable(
    refit_mesh: trimesh.Trimesh,
    wearable_mesh: trimesh.Trimesh,
    directions: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_offsets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """How far the template has to move to clear a wearable, asked from the wearable's side.

    The wearable's surface is sampled and each sample is credited to one
    template vertex, so no ray ever finds material that is not above it (a
    chin's normal points at a collar). The owner is the vertex whose outward
    line the sample sits on, found by walking the sample down the direction
    field from its nearest vertex: nearest alone is wrong inside a groove, where
    the far sheet of a fold is nearer to the ridges than to the floor, and the
    floor read 2 cm where its neighbours read 9.

    Args:
        refit_mesh: The template, at bind pose.
        wearable_mesh: The geometry to clear, in the same space.
        directions: Per template vertex outward direction, already relaxed.
        neighbor_indices: Flat CSR neighbour list over the template's own edges.
        neighbor_offsets: Per-vertex offsets into it, length N + 1.

    Returns:
        A tuple of (heights, covered): the travel per template vertex, and the
        template vertices the wearable is above.
    """
    # 1. Sample the wearable surface, densely enough for the template's spacing
    per_vertex_area = float(refit_mesh.area) / max(len(refit_mesh.vertices), 1)
    count = int(np.ceil(_PATCH_OVERSAMPLE * float(wearable_mesh.area) / max(per_vertex_area, 1e-12)))
    count = max(count, len(refit_mesh.vertices))
    samples = np.asarray(wearable_mesh.sample(count), dtype=np.float64)
    vertices = np.asarray(refit_mesh.vertices, dtype=np.float64)
    tree = cKDTree(vertices)

    def miss(owner):
        offset = samples - vertices[owner]
        along = np.einsum('ij,ij->i', offset, directions[owner])
        return np.linalg.norm(offset - directions[owner] * along[:, None], axis=1), along

    # 2. Owner of each sample: the vertex whose line passes closest, reached by
    # walking the sample down the current owner's direction until nothing moves.
    # A sample migrates at most its own standoff, so nothing leaks across limbs.
    _, nearest = tree.query(samples, k=1)
    sideways, heights = miss(nearest)
    for _ in range(8):
        feet = samples - directions[nearest] * np.maximum(heights, 0.0)[:, None]
        _, candidate = tree.query(feet, k=1)
        candidate_sideways, candidate_heights = miss(candidate)
        better = candidate_sideways < sideways
        if not np.any(better):
            break
        nearest = np.where(better, candidate, nearest)
        sideways = np.where(better, candidate_sideways, sideways)
        heights = np.where(better, candidate_heights, heights)

    # 3. Height per vertex: the furthest sample on its line, along its own direction
    result = np.zeros(len(vertices))
    np.maximum.at(result, nearest, np.einsum('ij,ij->i', samples - vertices[nearest], directions[nearest]))
    result = np.maximum(result, 0.0)

    # 4. Close the coverage so the floor of a groove the wearable bridges counts,
    # then fill the holes from the neighbours that won a sample, ring by ring
    covered = result > 0.0
    close_rings = int(round(_PATCH_CLOSE_RATIO * float(refit_mesh.scale)
                            / max(float(np.median(refit_mesh.edges_unique_length)), 1e-12)))
    if close_rings > 0:
        owners = np.repeat(np.arange(len(vertices)), np.diff(neighbor_offsets))

        def grow(mask, rings):
            for _ in range(rings):
                grown = mask.copy()
                np.logical_or.at(grown, owners, mask[neighbor_indices])
                mask = grown
            return mask

        covered = ~grow(~grow(covered, close_rings), close_rings)

        filled = result > 0.0
        for _ in range(2 * close_rings):
            holes = covered & ~filled
            if not np.any(holes):
                break
            totals = np.zeros(len(vertices))
            counts = np.zeros(len(vertices))
            np.add.at(totals, owners, np.where(filled, result, 0.0)[neighbor_indices])
            np.add.at(counts, owners, filled[neighbor_indices].astype(np.float64))
            spread = holes & (counts > 0.0)
            if not np.any(spread):
                break
            result[spread] = totals[spread] / counts[spread]
            filled |= spread

    # 5. The exact point-to-triangle distance as a floor, gated by the coverage
    # so a chin is never dragged to the collar below it
    closest, _, _ = trimesh.proximity.closest_point(wearable_mesh, vertices)
    gathered = np.einsum('ij,ij->i', closest - vertices, directions)
    result = np.maximum(np.where(covered, np.maximum(result, gathered), 0.0), 0.0)

    # 6. The clearance the patch stands proud of the wearable
    covered = result > 0.0
    result = np.where(covered, result + _PATCH_CLEARANCE_RATIO * float(refit_mesh.scale), 0.0)
    logger.info(f"Scattered {count} wearable samples onto {int(np.count_nonzero(covered))}/"
                f"{len(refit_mesh.vertices)} template vertices; height median "
                f"{float(np.median(result[covered])) if np.any(covered) else 0.0:.5f}, max {float(result.max()):.5f}.")

    return result, covered


def _clip_convergence(
    origins: np.ndarray,
    directions: np.ndarray,
    offsets: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_offsets: np.ndarray,
    min_ratio: float,
    max_passes: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Holds every displaced template edge above a fraction of its own length.

    Two vertices whose directions converge walk into each other once lifted far
    enough, and the faces around them fold (the tops of the inner thighs, at the
    crotch, came down to 0.09 of their edge). Each short edge has the travel of
    the end closing it scaled so the edge sits back on the bound, the other
    end's share taken as it stands. Only the closing end: scaling both lets a
    neighbour edge's correction reopen this one, and the passes never settle.

    Args:
        origins: (N, 3) template vertex positions.
        directions: (N, 3) unit travel directions.
        offsets: (N,) travel along them.
        neighbor_indices: Flat CSR neighbour list over the template's own edges.
        neighbor_offsets: Per-vertex offsets into it, length N + 1.
        min_ratio: Shortest an edge may become, as a fraction of its length. 0 disables.
        max_passes: Cap on the passes. Reaching it is logged, never silent.

    Returns:
        A tuple of (offsets, held): the travel, never larger than it came in,
        and the vertices that were held back (handed to the mush with the folds).
    """
    if min_ratio <= 0.0 or len(neighbor_indices) == 0:
        return offsets, np.zeros(len(offsets), dtype=bool)

    # 1. Every edge once, with how much each end's travel closes it per unit
    # (negative is towards the other end)
    owners = np.repeat(np.arange(len(offsets)), np.diff(neighbor_offsets))
    once = owners < neighbor_indices
    a, b = owners[once], neighbor_indices[once]
    edge = origins[b] - origins[a]
    length_sq = np.einsum('ij,ij->i', edge, edge)
    rate_a = -np.einsum('ij,ij->i', directions[a], edge)
    rate_b = np.einsum('ij,ij->i', directions[b], edge)
    target = (min_ratio - 1.0) * length_sq

    # 2. Scale the closing ends of the short edges until no edge is short. A
    # corrected edge sits exactly on the bound, hence the tolerance.
    clipped = offsets.astype(np.float64, copy=True)
    used_passes = 0
    for used_passes in range(1, max_passes + 1):
        share_a = clipped[a] * rate_a
        share_b = clipped[b] * rate_b
        short = share_a + share_b < target - 1e-6 * length_sq
        if not np.any(short):
            break
        factor = np.ones(len(offsets))
        for end, share, other in ((a, share_a, share_b), (b, share_b, share_a)):
            closing = short & (share < 0.0)
            scale = np.where(other[closing] < 0.0,
                             target[closing] / np.minimum(share[closing] + other[closing], -1e-12),
                             (target[closing] - other[closing]) / np.minimum(share[closing], -1e-12))
            np.minimum.at(factor, end[closing], np.clip(scale, 0.0, 1.0))
        clipped *= factor
    else:
        logger.warning(f"Convergence clip did not settle in {max_passes} passes.")

    # 3. Report who was held
    held = offsets - clipped > 1e-9
    if np.any(held):
        logger.info(f"Held {int(np.count_nonzero(held))} vertices back from their neighbours, by up to "
                    f"{float((offsets - clipped).max()):.5f}, in {used_passes} passes.")
    return clipped, held


def _lift_clear(
    field,
    positions: np.ndarray,
    directions: np.ndarray,
    raw_positions: np.ndarray,
    pitch: float
) -> Tuple[np.ndarray, int, float]:
    """Lifts the vertices the relaxation left inside the garment back out, by the least distance.

    Half a voxel at a time along each vertex's own direction until the field
    reads air, and never past the scatter position. Restoring the scatter
    position instead was where every spike on the patch came from: it is a max
    over samples, an outlier by construction.

    Args:
        field: The containment field of the wearable.
        positions: (N, 3) current patch positions.
        directions: (N, 3) unit travel directions.
        raw_positions: (N, 3) where the scatter put each vertex, the furthest it may go.
        pitch: The field's voxel pitch, for the step.

    Returns:
        A tuple of (positions, count, median lift).
    """
    # 1. Who is inside (tested past the isovalue, so float noise does not count)
    inside = field.sample(positions) >= field.level + 0.05
    if not np.any(inside):
        return positions, 0, 0.0

    index = np.flatnonzero(inside)
    start, heading = positions[index], directions[index]
    limit = np.einsum('ij,ij->i', raw_positions[index] - start, heading)

    # 2. March outward, half a voxel per step, until air or the scatter position
    step = 0.5 * pitch
    travelled = np.zeros(len(index))
    live = limit > 0.0
    while np.any(live):
        travelled[live] += step
        reached = live & (travelled >= limit)
        travelled[reached] = limit[reached]
        live &= ~reached
        if np.any(live):
            probe = start[live] + heading[live] * travelled[live, None]
            live[np.flatnonzero(live)[field.sample(probe) < field.level + 0.05]] = False

    # 3. Write the lifted positions back; past the limit, the scatter position is the answer
    lifted = start + heading * travelled[:, None]
    lifted[travelled >= limit] = raw_positions[index][travelled >= limit]
    positions = positions.copy()
    positions[index] = lifted
    return positions, int(len(index)), float(np.median(travelled))


def bake_anchor_patch(
    refit_mesh: trimesh.Trimesh,
    wearable_mesh: trimesh.Trimesh,
    bridge_ratio: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Displaces the template vertices a wearable sits on, out past its bulk.

    Produces the collision proxy a later layer refits against: the template
    keeps its topology and only the covered vertices move, each along a relaxed
    direction until it clears the wearable. The result only has to be smooth
    and generous -- a patch slightly too fat gives the next layer more air,
    while a dent or a step in it prints straight through.

    Args:
        refit_mesh: The template geometry to displace, at bind pose.
        wearable_mesh: The geometry to clear, in the same space.
        bridge_ratio: Opening welded shut in the containment field, in voxels.
            Zero for anything made in one piece; a chain or a beaded strap needs
            it to read as one volume.

    Returns:
        A tuple of (sparse_indices, positions): the displaced template vertex
        indices, ascending, and their new positions in the template's space.
    """
    start_time = time.time()

    # 1. The containment field, on the template's pitch: it only serves the
    # lifts below, which ask whether a vertex sits inside the bare material
    pitch = _PATCH_VOXEL_RATIO * float(refit_mesh.scale)
    logger.info(f"Patch pitch {pitch:.5f}.")
    field = tm_build_solid_field(mesh=wearable_mesh, pitch=pitch, bridge_gap=bridge_ratio * pitch, margin=0.0)

    # 2. The template's own edges (no proximity links: a body's limbs must not be tied together)
    origins = np.asarray(refit_mesh.vertices, dtype=np.float64)
    graph = _pack_smoothing_graph(refit_mesh, proximity_ratio=0.0)
    neighbor_indices = np.asarray(graph["neighborIndices"], dtype=np.int64)
    neighbor_offsets = np.asarray(graph["neighborOffsets"], dtype=np.int64)

    # 3. Travel directions: vertex normals relaxed over a distance, so that a
    # concavity's converging normals become locally parallel
    directions = np.asarray(refit_mesh.vertex_normals, dtype=np.float64)
    direction_iterations = int(round(_PATCH_DIRECTION_REACH_RATIO * float(refit_mesh.scale)
                                     / max(float(np.median(refit_mesh.edges_unique_length)), 1e-12)))
    if direction_iterations > 0:
        directions = _smooth_deformation_deltas(directions.astype(np.float32), neighbor_indices,
                                                neighbor_offsets, direction_iterations).astype(np.float64)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    # 4. Heights from the wearable's side, then the convergence bound
    offsets, covered = _patch_heights_from_wearable(refit_mesh, wearable_mesh, directions,
                                                    neighbor_indices, neighbor_offsets)
    offsets, clipped = _clip_convergence(origins, directions, offsets, neighbor_indices,
                                         neighbor_offsets, _PATCH_MIN_EDGE_RATIO)
    displacements = directions * offsets[:, None]

    # 5. Feather, with residual recovery so the tall parts (a hood) do not sink
    if _PATCH_FEATHER_ITERATIONS > 0:
        displacements = _smooth_deformation_deltas(
            displacements.astype(np.float32), neighbor_indices, neighbor_offsets,
            _PATCH_FEATHER_ITERATIONS, unshrink=1.0).astype(np.float64)

    # 6. Containment: whatever the feather sank into the wearable is lifted back out
    positions = origins + displacements
    raw_positions = origins + directions * offsets[:, None]
    positions, sank, lift = _lift_clear(field, positions, directions, raw_positions, pitch)
    if sank:
        logger.info(f"Lifted {sank} vertices the feather sank into the wearable back out, by a median of {lift:.4f}.")

    # 7. Delta mush over the folded faces and the held vertices, then containment again
    if _PATCH_MUSH_ITERATIONS > 0:
        patched = refit_mesh.copy()
        patched.vertices = positions
        folded = np.einsum("ij,ij->i", patched.face_normals, refit_mesh.face_normals) < 0.0
        suspect = np.unique(np.concatenate([refit_mesh.faces[folded].ravel().astype(np.int64),
                                            np.flatnonzero(clipped)]))
        if len(suspect):
            mushed = tm_delta_mush(deformed_mesh=patched, ref_mesh=refit_mesh, iterations=_PATCH_MUSH_ITERATIONS,
                                   mask_indices=suspect, falloff_distance=_PATCH_MUSH_FALLOFF_RATIO * float(refit_mesh.scale))
            positions = np.asarray(mushed.vertices, dtype=np.float64)
            positions, sank_again, lift = _lift_clear(field, positions, directions, raw_positions, pitch)
            if sank_again:
                logger.info(f"Lifted {sank_again} vertices the mush dragged into the wearable back out, by a median of {lift:.4f}.")
            inside = int(np.count_nonzero(field.sample(positions) >= field.level + 0.05))
            logger.info(f"Delta mushed {len(suspect)} vertices across {int(np.count_nonzero(folded))} folded faces "
                        f"and {int(np.count_nonzero(clipped))} clipped ones; {inside} left inside the wearable.")

    # 8. Keep only the vertices that actually moved
    sparse_indices = np.flatnonzero(np.linalg.norm(positions - origins, axis=1) > 0.01 * pitch)
    logger.info(f"Finished bake_anchor_patch in {time.time() - start_time:.4f} seconds: "
                f"{len(sparse_indices)}/{len(origins)} vertices displaced.")
    return sparse_indices, positions[sparse_indices]


def pack_anchor_patch(
    refit_mesh: trimesh.Trimesh,
    sparse_indices: np.ndarray,
    positions: np.ndarray,
    max_influences: int = 4
) -> AnchorSurfacePatch:
    """Anchors baked patch positions to the template so they track the avatar.

    Each patch point is anchored to the faces meeting at the very vertex it
    came from, with one-hot barycentrics: the blended anchor point is the bind
    vertex itself and the gap reduces to the displacement, which stretches with
    the body the same way a wearable's gaps do.

    Args:
        refit_mesh: The template geometry, at bind pose.
        sparse_indices: Displaced template vertex indices, as baked.
        positions: Their new positions, in the template's space.
        max_influences: Maximum incident faces kept per point, largest first.

    Returns:
        AnchorSurfacePatch: The patch payload, faceIds indexing the template's full face list.
    """
    sparse_indices = np.asarray(sparse_indices, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)

    # 1. The incident faces of each vertex, largest first so the slivers are the ones dropped
    incident = np.atleast_2d(refit_mesh.vertex_faces[sparse_indices])
    areas = np.where(incident >= 0, refit_mesh.area_faces[np.maximum(incident, 0)], -1.0)
    order = np.argsort(-areas, axis=1)[:, :max_influences]
    faces = np.take_along_axis(incident, order, axis=1)
    weights = np.take_along_axis(areas, order, axis=1)

    # 2. Pad short rings to a fixed width and normalize the area weights
    pad = max_influences - faces.shape[1]
    if pad > 0:
        faces = np.pad(faces, ((0, 0), (0, pad)), constant_values=-1)
        weights = np.pad(weights, ((0, 0), (0, pad)), constant_values=-1.0)
    faces = np.where(weights > 0.0, faces, -1)
    weights = np.maximum(weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    # 3. One-hot barycentrics: 1 at the corner the point came from, on every face of the ring
    corners = refit_mesh.faces[np.maximum(faces, 0)]
    barys = (corners == sparse_indices[:, None, None]).astype(np.float64)
    mapping = BarycentricMapping(face_indices=faces, barycentric_coords=barys,
                                 influence_weights=weights, valid_mask=weights.sum(axis=1) > 0.0)

    return {"positions": positions.tolist(), "sparseIndices": sparse_indices.tolist(),
            "vertexAnchors": _pack_anchor_set(positions, mapping, refit_mesh, max_influences)}


# =============================================================================
# PAYLOAD READERS
# =============================================================================
# Everything the layering needs is read off the payloads alone, at bind pose:
# the anchors as arrays, the anchor point and normal of each wearable vertex,
# and a few fields over the template vertices.

class _Anchors:
    """A payload's vertex anchors as contiguous arrays, read once.

    Attributes:
        face_ids: (A,) anchor surface row of each anchor.
        barys: (A, 3) barycentric coordinates, unclamped.
        weights: (A,) blend weights, pre-normalized per point.
        owner: (A,) the point each anchor belongs to.
        offsets: (N + 1,) CSR offsets; num_points is N.
        source: (S,) template face of each anchor surface row.
        corners: (S, 3, 3) bind triangles of the anchor surface.
        gaps: (N, 3) bind gap vectors.
    """

    def __init__(self, mapping: RuntimeMeshAnchor, anchor_set: Optional[PointAnchorSet] = None):
        # 1. The anchors of the point set (the wearable's by default)
        anchor_set = mapping["vertexAnchors"] if anchor_set is None else anchor_set
        anchors = anchor_set.get("anchors", [])
        self.offsets = np.asarray(anchor_set["anchorOffsets"], dtype=np.int64)
        self.num_points = max(len(self.offsets) - 1, 0)
        self.empty = not anchors or self.num_points == 0 or not mapping["anchorSurface"]
        self.face_ids = np.fromiter((a["faceId"] for a in anchors), dtype=np.int64)
        self.barys = np.array([a["baryCoords"] for a in anchors], dtype=np.float64).reshape(-1, 3)
        self.weights = np.fromiter((a["weight"] for a in anchors), dtype=np.float64)
        self.owner = np.repeat(np.arange(self.num_points), np.diff(self.offsets)) if self.num_points else np.zeros(0, dtype=np.int64)

        # 2. The anchor surface they address, and the gaps
        surface = mapping["anchorSurface"]
        self.source = np.array([f["sourceFaceId"] for f in surface], dtype=np.int64)
        self.corners = np.array([[f["point0"], f["point1"], f["point2"]] for f in surface], dtype=np.float64).reshape(-1, 3, 3)
        self.gaps = np.asarray(anchor_set.get("gapVectors", []), dtype=np.float64).reshape(-1, 3)

    def blend(self, per_anchor: np.ndarray) -> np.ndarray:
        """Sums a per-anchor quantity into its points, weighted by the anchors' weights."""
        shape = (self.num_points,) + per_anchor.shape[1:]
        out = np.zeros(shape)
        if len(per_anchor):
            np.add.at(out, self.owner, per_anchor * self.weights.reshape(-1, *([1] * (per_anchor.ndim - 1))))
        return out


def _surface_orientation(corners: np.ndarray, gaps: np.ndarray, normals: np.ndarray) -> float:
    """Whether the bind normals turn outward, +1 or -1.

    A payload converted to a left-handed space (the runtime negates x) is
    mirrored, and a mirror reverses every cross product: read as authored, the
    surface's normals would point into the body and every height below would
    come out negative. A wearable sits outside the body, so its own gaps are the
    answer: they lean along the outward normals, by a wide margin on every asset
    measured (99 % of the vertices agree). A wearable baked skin-tight has no
    gaps to vote with, and the faces vote against the direction from the
    surface's centre to their own instead, which only holds where the surface
    wraps its own centre.

    Args:
        corners: (F, 3, 3) bind triangles of the anchor surface.
        gaps: (N, 3) bind gap vectors.
        normals: (N, 3) blended bind normals, as the winding gives them.
    """
    # 1. The gaps vote: a wearable is outside the body
    if len(gaps) == len(normals):
        vote = float(np.einsum('ij,ij->i', gaps, normals).sum())
        if abs(vote) > _ORIENTATION_MARGIN * float(np.linalg.norm(gaps, axis=1).sum()):
            return -1.0 if vote < 0.0 else 1.0

    # 2. Nothing to vote with: the winding against the surface's own centre
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = np.einsum('ij,ij->i', cross, corners.mean(axis=1) - corners.reshape(-1, 3).mean(axis=0))
    return -1.0 if np.sign(outward).dot(np.linalg.norm(cross, axis=1)) < 0.0 else 1.0


def _anchor_frames(mapping: RuntimeMeshAnchor) -> Tuple[np.ndarray, np.ndarray]:
    """Where each wearable vertex is bound to the template, and which way is out.

    The blended bind surface point (where the vertex converges when its gap goes
    to zero) and the blended bind normal of the faces it is anchored to, turned
    outward whatever the handedness of the space the payload lives in.

    Returns:
        A tuple of (surface_points, normals), one row per wearable vertex.
    """
    a = _Anchors(mapping)
    if a.empty:
        return np.zeros((a.num_points, 3)), np.zeros((a.num_points, 3))

    # 1. Bind face normals of the anchor surface
    normals = np.cross(a.corners[:, 1] - a.corners[:, 0], a.corners[:, 2] - a.corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    # 2. Blend points and normals through the anchors
    points = a.blend(np.einsum('nc,ncd->nd', a.barys, a.corners[a.face_ids]))
    blended = a.blend(normals[a.face_ids])
    blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-12)

    # 3. Turned outward, whatever the handedness of the space the payload lives in
    return points, blended * _surface_orientation(a.corners, a.gaps, blended)


def _anchor_standoff(mapping: RuntimeMeshAnchor) -> np.ndarray:
    """How far each vertex's anchor point itself stands off the template surface.

    The planar projection hands out barycentrics outside [0, 1], and a point
    blended with those lies on the face's plane extended: outside a convex
    body, inside a concave one. A boot vertex on the shin read 0.63 along its
    normal and stood 1.09 from the body. Measured as the signed distance from
    the raw anchor point to the payload's own bind triangles, so it is zero
    wherever the anchors are honest and needs nothing but the payload.
    """
    a = _Anchors(mapping)
    points, normals = _anchor_frames(mapping)
    if a.empty:
        return np.zeros(a.num_points)

    # 1. The bind triangles as a soup, 2. the closest point on it, signed by the normal
    soup = trimesh.Trimesh(vertices=a.corners.reshape(-1, 3),
                           faces=np.arange(len(a.corners) * 3).reshape(-1, 3), process=False)
    closest, _, _ = trimesh.proximity.closest_point(soup, points)
    return np.einsum('ij,ij->i', points - closest, normals)


def _gap_heights(mapping: RuntimeMeshAnchor) -> np.ndarray:
    """How far each wearable vertex stands off the template surface, along its anchor normal.

    The gap's normal component plus the anchor point's own standoff, so two
    garments' heights are compared from the surface and not from anchor points
    that float off it by different amounts.
    """
    _, normals = _anchor_frames(mapping)
    gaps = _Anchors(mapping).gaps
    if len(gaps) != len(normals):
        return np.zeros(len(normals))
    return np.einsum('ij,ij->i', gaps, normals) + _anchor_standoff(mapping)


def _read_surface_field(
    refit_mesh: trimesh.Trimesh,
    mapping: RuntimeMeshAnchor,
    field: np.ndarray,
    normalize: bool
) -> np.ndarray:
    """Reads a per-template-vertex quantity at each wearable vertex, through its own anchors.

    Interpolating through the anchors the vertex is already bound by keeps the
    read free of any radius to pick. Barycentrics are clamped to [0, 1]: a face
    reached from outside must not lend what the vertex has no share of.

    Args:
        refit_mesh: The template, for its faces.
        mapping: The wearable's payload, for its anchors.
        field: One value per template vertex.
        normalize: Divide each anchor's blend by its clamped barycentric sum.
            On for a height, off for a mask (whose sum is the answer).

    Returns:
        One value per wearable vertex.
    """
    a = _Anchors(mapping)
    if a.empty:
        return np.zeros(a.num_points)

    # 1. Per anchor: the field at the face's corners, blended with clamped barycentrics
    clamped = np.clip(a.barys, 0.0, 1.0)
    per_anchor = np.einsum('nc,nc->n', clamped, field[refit_mesh.faces[a.source[a.face_ids]]])
    if normalize:
        per_anchor /= np.maximum(clamped.sum(axis=1), 1e-12)

    # 2. Per vertex: the weighted sum over its anchors
    return a.blend(per_anchor)


def _anchor_lift(bare_mesh: trimesh.Trimesh, patched_mesh: trimesh.Trimesh, mapping: RuntimeMeshAnchor) -> np.ndarray:
    """How far the patches below lifted each vertex's anchor point, along its normal.

    Measured, not estimated: the blended surface point on the patched template
    minus the same on the bare one, with the raw barycentrics the evaluation
    uses. A field read with clamped barycentrics was a centimetre short under a
    vertex anchored past its faces' edges -- the evaluation extrapolates, so the
    ground is measured the way it extrapolates.
    """
    a = _Anchors(mapping)
    if a.empty:
        return np.zeros(a.num_points)

    # 1. The template's displacement at each anchor face, blended with the raw barycentrics
    moved = np.asarray(patched_mesh.vertices, dtype=np.float64) - np.asarray(bare_mesh.vertices, dtype=np.float64)
    lift = a.blend(np.einsum('nc,ncd->nd', a.barys, moved[bare_mesh.faces[a.source[a.face_ids]]]))

    # 2. Its component along the anchor normal
    _, normals = _anchor_frames(mapping)
    return np.einsum('ij,ij->i', lift, normals)


def _deformation_gradients(
    mapping: RuntimeMeshAnchor,
    basis_mesh: trimesh.Trimesh
) -> Tuple[np.ndarray, np.ndarray]:
    """Per anchor surface face, the affine gradient from its bind triangle to the driven one.

    Unnormalized edges and a unit normal, so gaps stretch and shear with the
    surface. The bind basis is rebuilt from the raw points every time, in the
    space the payload currently lives in: a pre-inverted basis does not survive
    a handedness conversion.

    Args:
        mapping: The payload, for its bind triangles and their source faces.
        basis_mesh: The driven template the current triangles are read from.

    Returns:
        A tuple of (transforms, bind_edge_lengths): (S, 3, 3) gradients, and the
        bind edge lengths that set the smoothing reach.
    """
    # 1. Bind triangles and the template faces they come from
    surface = mapping["anchorSurface"]
    p0 = np.array([f["point0"] for f in surface], dtype=np.float32)
    p1 = np.array([f["point1"] for f in surface], dtype=np.float32)
    p2 = np.array([f["point2"] for f in surface], dtype=np.float32)
    source = np.array([f["sourceFaceId"] for f in surface], dtype=np.int64)

    def basis(a, b, c):
        x, y = b - a, c - a
        z = np.cross(x, y)
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        return np.stack([x, y, z], axis=-1)

    # 2. Driven basis times the inverted bind basis
    driven = basis_mesh.vertices[basis_mesh.faces[source]]
    transforms = np.matmul(basis(driven[:, 0], driven[:, 1], driven[:, 2]), np.linalg.pinv(basis(p0, p1, p2)))
    lengths = np.concatenate([np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p0, axis=1),
                              np.linalg.norm(p2 - p1, axis=1)])
    return transforms, lengths


# =============================================================================
# LOAD AND APPLY FUNCTIONS
# =============================================================================

def _unpack_anchor_set(
    anchor_set: PointAnchorSet,
    rotation_matrices: np.ndarray,
    deformed_triangles: np.ndarray,
    num_points: int,
    bind_rotations: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates anchored points: the blended surface point plus the gap carried by the blended gradient.

    Trusts the pre-normalized bake weights to bypass runtime division. Joints
    also get a rotation, orthogonalized by SVD so bones never shear.
    """
    anchors = anchor_set.get("anchors", [])
    if not anchors or num_points == 0:
        return np.zeros((num_points, 3)), np.zeros((num_points, 3, 3))

    # 1. Unpack the flattened JSON dictionaries into contiguous arrays
    face_ids = np.fromiter((a["faceId"] for a in anchors), dtype=np.int32)
    barys = np.array([a["baryCoords"] for a in anchors], dtype=np.float32)
    weights = np.fromiter((a["weight"] for a in anchors), dtype=np.float32)
    offsets = np.array(anchor_set["anchorOffsets"], dtype=np.int32)
    gaps = np.array(anchor_set["gapVectors"], dtype=np.float32)

    # 2. Surface points on the deformed triangles
    surface_pts = np.sum(barys[..., None] * deformed_triangles[face_ids], axis=1)

    # 3. Accumulate the weighted positions and gradients per point (weights pre-sum to 1)
    pt_indices = np.repeat(np.arange(num_points), np.diff(offsets))
    blended_pos = np.zeros((num_points, 3), dtype=np.float32)
    blended_mat = np.zeros((num_points, 3, 3), dtype=np.float32)
    np.add.at(blended_pos, pt_indices, surface_pts * weights[:, None])
    np.add.at(blended_mat, pt_indices, rotation_matrices[face_ids] * weights[:, None, None])

    # 4. Identity fallback for unmapped points
    valid = np.bincount(pt_indices, minlength=num_points) > 0
    blended_mat[~valid] = np.eye(3)

    # 5. Position: blended surface point plus the gap carried by the blended gradient
    new_positions = blended_pos.copy()
    new_positions[valid] = blended_pos[valid] + np.einsum('nij,nj->ni', blended_mat[valid], gaps[valid])

    # 6. Joint rotations: SVD orthogonalization, reflections corrected
    new_rotations = np.zeros((num_points, 3, 3))
    if bind_rotations is not None:
        U, _, Vh = np.linalg.svd(blended_mat)
        ortho = np.matmul(U, Vh)
        flip = np.linalg.det(ortho) < 0
        if np.any(flip):
            U[flip, :, 2] *= -1
            ortho[flip] = np.matmul(U[flip], Vh[flip])
        new_rotations = np.matmul(ortho, bind_rotations)

    return new_positions, new_rotations


def apply_anchor_patch(
    refit_deformed_mesh: trimesh.Trimesh,
    anchor_mapping: RuntimeMeshAnchor,
    patch: AnchorSurfacePatch,
    basis_mesh: Optional[trimesh.Trimesh] = None
) -> np.ndarray:
    """Evaluates a patch against a deformed template, by the same arithmetic as a wearable.

    Args:
        refit_deformed_mesh: The driven template, with whatever patches are
            already written into it: where the anchor points land.
        anchor_mapping: The payload the patch belongs to, for its anchor surface.
        patch: The patch payload to evaluate.
        basis_mesh: The template the deformation gradient is read from. The bare
            driver when this patch is stacked on another one, so a patch is a
            lift for whatever sits on it and never a stretch. None reads it from
            refit_deformed_mesh.

    Returns:
        np.ndarray: Deformed positions for the patch's sparseIndices, in order.
    """
    # 1. Gradients off the basis mesh, 2. surface points off the patched one, 3. evaluate
    transforms, _ = _deformation_gradients(anchor_mapping, basis_mesh or refit_deformed_mesh)
    source = np.array([f["sourceFaceId"] for f in anchor_mapping["anchorSurface"]], dtype=np.int64)
    tri_verts = refit_deformed_mesh.vertices[refit_deformed_mesh.faces[source]]
    positions, _ = _unpack_anchor_set(patch["vertexAnchors"], transforms, tri_verts, len(patch["sparseIndices"]))
    return positions


def convert_anchor_mapping_space(
    anchor_mapping: RuntimeMeshAnchor,
    per_axis_scale: Tuple[float, float, float]
) -> RuntimeMeshAnchor:
    """Converts the payload's spatial data between coordinate conventions.

    A per-axis diagonal scale (units and/or mirroring) applied to every point
    and vector: the anchor surface's bind points, the patches' positions and
    gaps, and both anchor sets' gaps. Barycentrics, weights, indices and the
    smoothing graph are invariant and pass through untouched.

    Args:
        anchor_mapping: Structured mapping payload. Modified in place.
        per_axis_scale: Diagonal (sx, sy, sz) applied to every point and vector.

    Returns:
        RuntimeMeshAnchor: The converted mapping payload (same object).
    """
    scale = np.asarray(per_axis_scale, dtype=np.float64)

    # 1. The anchor surface's bind points
    for f in anchor_mapping["anchorSurface"]:
        for key in ("point0", "point1", "point2"):
            f[key] = (np.asarray(f[key], dtype=np.float64) * scale).tolist()

    # 2. The patches' positions and gaps
    for source in anchor_mapping.get("anchorSurfaceSources", []):
        patch = source.get("patch")
        if patch:
            if patch["positions"]:
                patch["positions"] = (np.asarray(patch["positions"], dtype=np.float64) * scale).tolist()
            if patch["vertexAnchors"]["gapVectors"]:
                patch["vertexAnchors"]["gapVectors"] = (
                    np.asarray(patch["vertexAnchors"]["gapVectors"], dtype=np.float64) * scale).tolist()

    # 3. The vertex and joint gaps
    for set_key in ("vertexAnchors", "jointAnchors"):
        gaps = anchor_mapping[set_key]["gapVectors"]
        if gaps:
            anchor_mapping[set_key]["gapVectors"] = (np.asarray(gaps, dtype=np.float64) * scale).tolist()

    return anchor_mapping


def reverse_anchor_surface_winding(anchor_mapping: RuntimeMeshAnchor) -> RuntimeMeshAnchor:
    """Reverses the corner order of every bind face, the way the serializer does.

    A mesh serialized for the runtime goes through GUPM's ``process_mesh_utility``,
    which mirrors the points **and** reverses the index buffer
    (``usd_utility.flip_winding_order``, ``(a, b, c) -> (c, b, a)``). The payload's
    bind faces are read straight off the authored USD, so one shipped unchanged
    addresses corner 0 where the runtime holds corner 2: the barycentrics land on
    the wrong corners and the deformation gradient comes out a reflection, which
    negates every gap's component along the normal and drops the wearable inside
    the body.

    The permutation has to be the serializer's own. Reversing the winding the
    other way round (``(a, c, b)``, swapping point1 and point2) turns the triangle
    over just the same and still addresses the wrong corners.

    Args:
        anchor_mapping: Structured mapping payload. Modified in place.

    Returns:
        RuntimeMeshAnchor: The reversed mapping payload (same object).
    """
    # 1. The bind faces, swapping the corners the serializer swaps
    for face in anchor_mapping["anchorSurface"]:
        face["point0"], face["point2"] = face["point2"], face["point0"]

    # 2. Every anchor set that addresses them, the barycentrics with the corners
    anchor_sets = [anchor_mapping["vertexAnchors"], anchor_mapping["jointAnchors"]]
    for source in anchor_mapping.get("anchorSurfaceSources", []):
        patch = source.get("patch")
        if patch:
            anchor_sets.append(patch["vertexAnchors"])

    for anchor_set in anchor_sets:
        for anchor in anchor_set["anchors"]:
            bary = anchor["baryCoords"]
            bary[0], bary[2] = bary[2], bary[0]

    return anchor_mapping


# =============================================================================
# LAYERING
# =============================================================================
# A stack of wearables, innermost first. Each one rides what the stack below it
# stands out of its own clearance, and stays under the inner surface of the one
# above it. Everything is settled once per combination, in bind heights read off
# the payloads, and applied as one gap correction per vertex and one amount per
# patch vertex; nothing here runs per frame.

def layer_stiffness_of(layer: AnchorLayer) -> float:
    """A layer's stiffness: the caller's override if given, else the payload's authored value."""
    if layer.get("layerStiffness") is not None:
        return float(np.clip(layer["layerStiffness"], 0.0, 1.0))
    return float(np.clip((layer.get("anchorMapping") or {}).get("layerStiffness", 0.0), 0.0, 1.0))


def layer_press_share(inner_stiffness: float, outer_stiffness: float) -> float:
    """The share of an overlap the inner garment gives up to the outer one.

    press = outer_stiffness x (1 - inner_stiffness): how hard the outer one
    pushes, times how far the inner one gives. The outer one rides the rest, so
    the overlap is paid exactly once, and a garment at 1 is never pressed.

    Args:
        inner_stiffness: The stiffness of the garment underneath, from 0 to 1.
        outer_stiffness: The stiffness of the garment on top, from 0 to 1.

    Returns:
        The inner garment's share of the overlap, from 0 to 1.
    """
    inner = float(np.clip(inner_stiffness, 0.0, 1.0))
    outer = float(np.clip(outer_stiffness, 0.0, 1.0))
    return outer * (1.0 - inner)


def _press_matrix(stack: Sequence[AnchorLayer]) -> np.ndarray:
    """press[inner, outer] for every pair of the stack, zero elsewhere."""
    stiffness = [layer_stiffness_of(layer) for layer in stack]
    press = np.zeros((len(stack), len(stack)))
    for inner in range(len(stack)):
        for outer in range(inner + 1, len(stack)):
            press[inner, outer] = layer_press_share(stiffness[inner], stiffness[outer])
    return press


def _first_patch(anchor_mapping: RuntimeMeshAnchor) -> Optional[AnchorSurfacePatch]:
    """The patch on the payload's first anchor surface source, if it carries one."""
    for source in anchor_mapping.get("anchorSurfaceSources", []):
        patch = source.get("patch")
        if patch and patch.get("sparseIndices"):
            return patch
    return None


def _template_adjacency(refit_mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """The template's own edges as (owner, neighbour) index pairs, both ways."""
    graph = _pack_smoothing_graph(refit_mesh, proximity_ratio=0.0)
    neighbor_indices = np.asarray(graph["neighborIndices"], dtype=np.int64)
    neighbor_offsets = np.asarray(graph["neighborOffsets"], dtype=np.int64)
    return np.repeat(np.arange(len(neighbor_offsets) - 1), np.diff(neighbor_offsets)), neighbor_indices


def _garment_edge(mapping: RuntimeMeshAnchor) -> float:
    """A garment's median edge at bind, off its payload: positions from the anchors, edges from the graph."""
    points, _ = _anchor_frames(mapping)
    a = _Anchors(mapping)
    if a.empty or len(a.gaps) != len(points):
        return 0.0

    # 1. Bind positions, 2. every graph link once, 3. the median length
    positions = points + a.gaps
    graph = mapping["smoothingGraph"]
    neighbor_indices = np.asarray(graph["neighborIndices"], dtype=np.int64)
    neighbor_offsets = np.asarray(graph["neighborOffsets"], dtype=np.int64)
    sources = np.repeat(np.arange(len(neighbor_offsets) - 1), np.diff(neighbor_offsets))
    once = sources < neighbor_indices
    if not np.any(once):
        return 0.0
    return float(np.median(np.linalg.norm(positions[sources[once]] - positions[neighbor_indices[once]], axis=1)))


def _landing_rings(refit_mesh: trimesh.Trimesh, mapping: RuntimeMeshAnchor) -> int:
    """How many template rings separate a garment's landings: its edge over the template's.

    Derived rather than set, for the reason every ring count became a
    distance: rings halved their reach the day the template went from 3498 to
    14034 points.
    """
    garment_edge = _garment_edge(mapping)
    template_edge = float(np.median(refit_mesh.edges_unique_length))
    if garment_edge <= 0.0:
        return 1
    return max(int(np.ceil(garment_edge / max(template_edge, 1e-12))), 1)


def _layer_clearance_field(
    refit_mesh: trimesh.Trimesh,
    mapping: RuntimeMeshAnchor,
    owners: np.ndarray,
    neighbor_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """How far above each template vertex a garment's inner surface sits, and where that counts.

    The lowest standoff of the garment's vertices landing around each template
    vertex -- lowest, so a panel with thickness reads its inner face. Defined
    only where the garment lands: an open front, a hem, a neckline read as no
    ceiling. Past the rim the weight fades to 0 over the landing rings, so a
    garment comes out from under another over a distance rather than a step.

    Args:
        refit_mesh: The template, for its faces and edges.
        mapping: The garment's payload.
        owners: Template adjacency, owner side (see _template_adjacency).
        neighbor_indices: Template adjacency, neighbour side.

    Returns:
        A tuple of (heights, weight): the height per template vertex, and how
        much it counts -- 1 under the garment, fading to 0 past its rim.
    """
    heights = np.maximum(_gap_heights(mapping), 0.0)
    a = _Anchors(mapping)
    field = np.full(len(refit_mesh.vertices), np.inf)
    if a.empty or len(heights) == 0:
        return field, np.zeros(len(refit_mesh.vertices))

    # 1. Lowest landing height per template vertex, over the corners of each anchor face
    corners = refit_mesh.faces[a.source[a.face_ids]]
    np.minimum.at(field, corners.ravel(), np.repeat(heights[a.owner], 3))
    defined = np.isfinite(field)

    def grow(mask, rings):
        for _ in range(rings):
            grown = mask.copy()
            np.logical_or.at(grown, owners, mask[neighbor_indices])
            mask = grown
        return mask

    # 2. Close the coverage over the landing rings: a groove the garment bridges
    # counts, a hem does not grow
    rings = _landing_rings(refit_mesh, mapping)
    closed = ~grow(~grow(defined, rings), rings)

    # 3. The fade past the rim, as many rings again
    weight = closed.astype(np.float64)
    reach = closed.copy()
    for ring in range(1, rings + 1):
        grown = grow(reach, 1)
        weight[grown & ~reach] = 1.0 - ring / (rings + 1.0)
        reach = grown

    # 4. Fill the holes and the fade from the mean of the covered neighbours
    # (the mean: a hole is most often a groove the garment bridges, where its
    # surface stands higher than over the ridges; the lowest neighbour pressed into it)
    while True:
        totals = np.zeros(len(field))
        counts = np.zeros(len(field))
        np.add.at(totals, owners, np.where(defined, field, 0.0)[neighbor_indices])
        np.add.at(counts, owners, defined[neighbor_indices].astype(np.float64))
        spread = reach & ~defined & (counts > 0.0)
        if not np.any(spread):
            break
        field[spread] = totals[spread] / counts[spread]
        defined |= spread

    return field, weight


def _layer_bulk_field(refit_mesh: trimesh.Trimesh, mapping: RuntimeMeshAnchor) -> np.ndarray:
    """How far a garment's patch lifts each template vertex, zero where it has none.

    A patch's anchors are one-hot on the template's own vertices, so its gap
    vector is the displacement itself and its length is the lift.
    """
    field = np.zeros(len(refit_mesh.vertices))
    patch = _first_patch(mapping)
    if patch is not None:
        gaps = np.asarray(patch["vertexAnchors"]["gapVectors"], dtype=np.float64)
        field[np.asarray(patch["sparseIndices"], dtype=np.int64)] = np.linalg.norm(gaps, axis=1)
    return field


class _StackFields:
    """Where every layer of a stack stands and what it stays under, per template vertex.

    A garment rises over the stack below it only by what that stack stands out
    of the garment's own clearance, and by (1 - press) of that: riding the whole
    bulk below counted every garment's slack twice. What is left of a garment's
    bulk depends on where the garments above end up, and where they end up on
    what is left below, so the two are settled by fixed point; the error
    shrinks by 1 - press per pass.

    Attributes, all (L, V):
        free: each layer's patch lift as baked.
        bulk: what is left of it under the layers above.
        lift: how far each layer rises over the stack below it.
        ground: the top of the stack below each layer.
        ceiling: the inner surface of the layer above, lifted and held under its own ceiling.
        defined, weight: where the ceiling means anything, and how much (fading past the rim).
    """

    def __init__(self, refit_mesh: trimesh.Trimesh, stack: Sequence[AnchorLayer]):
        count, size = len(stack), len(refit_mesh.vertices)
        owners, neighbor_indices = _template_adjacency(refit_mesh)
        self.press = _press_matrix(stack)
        self.gap = _PATCH_CLEARANCE_RATIO * float(refit_mesh.scale)

        # 1. Per layer: its patch lift and its inner surface over the template
        self.free = np.zeros((count, size))
        clearance = np.full((count, size), np.inf)
        weight = np.zeros((count, size))
        for index, layer in enumerate(stack):
            mapping = layer.get("anchorMapping")
            if mapping is not None:
                self.free[index] = _layer_bulk_field(refit_mesh, mapping)
                clearance[index], weight[index] = _layer_clearance_field(refit_mesh, mapping, owners, neighbor_indices)
        covered = weight > 0.0

        self.bulk = self.free.copy()
        self.lift = np.zeros((count, size))
        self.ground = np.zeros((count, size))
        self.ceiling = np.full((count, size), np.inf)
        self.defined = np.zeros((count, size), dtype=bool)
        self.weight = np.zeros((count, size))

        # 2. Settle by fixed point, from the free bulk down
        for _ in range(16):
            previous = self.bulk

            # 2a. Bottom-up: what each layer rides of the stack below it
            top = np.zeros(size)
            stacked = np.zeros(size, dtype=bool)
            for index in range(count):
                overlap = np.where(stacked & covered[index], np.maximum(top + self.gap - clearance[index], 0.0), 0.0)
                press = self.press[index - 1, index] if index > 0 else 0.0
                self.lift[index] = (1.0 - press) * overlap
                self.ground[index] = top
                has = self.free[index] > 0.0
                top = np.where(has, np.maximum(top, self.bulk[index] + self.lift[index]), top)
                stacked |= has

            # 2b. Top-down: the inner surface of the layer above, one clearance per storey
            self.ceiling[-1], self.defined[-1], self.weight[-1] = np.inf, False, 0.0
            for index in range(count - 2, -1, -1):
                above = index + 1
                inner = np.where(covered[above], clearance[above] + self.lift[above], np.inf)
                inner = np.where(self.defined[above], np.minimum(inner, self.ceiling[above] - self.gap), inner)
                self.ceiling[index] = inner
                self.defined[index] = covered[above] | self.defined[above]
                self.weight[index] = np.maximum(weight[above], self.weight[above])

            # 2c. What is left of each layer's bulk under that, by the ceiling's weight
            clipped = np.maximum(np.minimum(self.free, self.ceiling - self.lift), 0.0)
            self.bulk = np.where(self.defined, self.free - self.weight * (self.free - clipped), self.free)
            if np.abs(self.bulk - previous).max() < 1e-3 * self.gap:
                break


def patch_refit_mesh(
    refit_deformed_mesh: trimesh.Trimesh,
    under_layers: Sequence[AnchorLayer],
    layer_stiffness: float = 0.0,
    over_layers: Sequence[AnchorLayer] = ()
) -> trimesh.Trimesh:
    """The template as this wearable sees it: the top of the stack below, written in.

    under_layers is the equipping order. Each patch takes the surface from
    where the previous one left it up to that garment's top -- what it has left
    under the layers above, lifted by what it rides -- and no further, at the
    share this wearable's stiffness does not press out of it. Gradients come
    off the bare driver, so a patch is a lift, never a stretch.

    Args:
        refit_deformed_mesh: The template, driven.
        under_layers: The wearables equipped below this one, innermost first.
        layer_stiffness: This wearable's own stiffness, from 0 to 1.
        over_layers: The wearables equipped above this one; only read to know
            how much of the ones below is still there to ride.

    Returns:
        A copy of the template with the patches written in. The input is untouched.
    """
    patched = refit_deformed_mesh.copy()
    if not under_layers:
        return patched

    # 1. Settle the whole stack, this wearable included
    stack = [*under_layers, {"layerStiffness": layer_stiffness}, *over_layers]
    fields = _StackFields(refit_deformed_mesh, stack)
    reader = len(under_layers)
    written = np.zeros(len(patched.vertices))

    # 2. Write each under-layer's patch, in equipping order, up to its top
    for index, layer in enumerate(under_layers):
        mapping = layer.get("anchorMapping")
        patch = _first_patch(mapping) if mapping is not None else None
        if patch is None:
            continue
        indices = np.asarray(patch["sparseIndices"], dtype=np.int64)
        free = fields.free[index, indices]
        top = np.minimum(fields.bulk[index, indices] + fields.lift[index, indices], fields.ground[reader, indices])
        amount = np.where(free > 0.0, np.maximum(top - written[indices], 0.0) / np.maximum(free, 1e-12), 0.0)
        amount *= 1.0 - fields.press[index, reader]
        if not np.any(amount > 0.0):
            continue

        deformed = apply_anchor_patch(patched, mapping, patch, basis_mesh=refit_deformed_mesh)
        vertices = np.asarray(patched.vertices, dtype=np.float64)
        vertices[indices] += amount[:, None] * (deformed - vertices[indices])
        patched.vertices = vertices
        written[indices] = np.maximum(written[indices], top)

    return patched


def _ceiling_excess(
    bare_mesh: trimesh.Trimesh,
    patched_mesh: trimesh.Trimesh,
    stack: Sequence[AnchorLayer],
    index: int
) -> Optional[np.ndarray]:
    """How much to take off each vertex's gap, along its anchor normal, for the stack.

    Two parts. The relief: what the patches below lifted the vertex beyond what
    the garment rides, read off the lift field so a panel's two faces move
    together. The excess: how far the vertex's top would still stand out past
    the ceiling the layer above leaves, faded by the ceiling's weight. Only what
    stands out is pressed; a garment that fits under the one over it is not
    touched, whatever the stiffnesses.

    Args:
        bare_mesh: The template as handed in, driven.
        patched_mesh: The same, with the layers below written in.
        stack: Every wearable equipped, innermost first, this one included.
        index: Which entry of the stack is this wearable.

    Returns:
        One length per vertex of this wearable, zero where nothing changes.
    """
    mapping = stack[index]["anchorMapping"]
    fields = _StackFields(bare_mesh, stack)
    defined = fields.defined[index]

    # 1. The ceiling less one clearance, read through the binding: the blend's
    # share landing under a ceiling (weighted by the fade), and the ceiling over that share
    ceiling = np.where(defined, fields.ceiling[index] - fields.gap, 0.0)
    fade = np.where(defined, fields.weight[index], 0.0)
    weight = _read_surface_field(bare_mesh, mapping, np.ones(len(ceiling)), normalize=False)
    covered = _read_surface_field(bare_mesh, mapping, fade, normalize=False)
    total = _read_surface_field(bare_mesh, mapping, ceiling * fade, normalize=False)
    under = covered > 1e-9
    coverage = np.where(under, covered / np.maximum(weight, 1e-9), 0.0).clip(0.0, 1.0)
    ceiling_here = np.where(under, total / np.maximum(covered, 1e-9), np.inf)

    # 2. The relief: the measured lift less what the garment rides
    lift = _anchor_lift(bare_mesh, patched_mesh, mapping)
    rides = _read_surface_field(bare_mesh, mapping, fields.lift[index], normalize=True)
    relief = np.maximum(lift - rides, 0.0)

    # 3. The excess above the ceiling, faded by the coverage
    top = lift - relief + _gap_heights(mapping)
    excess = coverage * np.maximum(top - ceiling_here, 0.0) + relief

    held = excess > 1e-6
    logger.info(f"Ceiling over {int(np.count_nonzero(under))}/{len(excess)} vertices: "
                f"{int(np.count_nonzero(held))} held back, by a median of "
                f"{float(np.median(excess[held])) if np.any(held) else 0.0:.4f} and at most {float(excess.max()):.4f}.")
    return excess


def apply_anchor_mapping(
    refit_deformed_mesh: trimesh.Trimesh,
    wearable_mesh: trimesh.Trimesh,
    anchor_mapping: RuntimeMeshAnchor,
    joint_bind_rotations: Optional[np.ndarray] = None,
    delta_smooth_iterations: int = 5,
    delta_smooth_unshrink: float = 1.0,
    layer_stiffness: Optional[float] = None,
    under_layers: Sequence[AnchorLayer] = (),
    over_layers: Sequence[AnchorLayer] = ()
) -> Tuple[trimesh.Trimesh, Optional[np.ndarray], Optional[np.ndarray]]:
    """Applies the anchor payload using the true affine deformation gradient.

    Alone or inside a stack: the layers below are written into the template
    before anything is read off it, and the gaps are shortened by whatever
    would stand out above the layers on top. A vertex is affine in its gap and
    the smoothing is linear, so the wearable is evaluated once however deep the
    stack is.

    Args:
        refit_deformed_mesh: The actively driven mesh geometry.
        wearable_mesh: The target mesh geometry to deform.
        anchor_mapping: Structured mapping payload.
        joint_bind_rotations: Optional initial joint orientations.
        delta_smooth_iterations: Diffusion passes on the deformation deltas over
            the baked wearable graph. 0 recovers the raw anchor evaluation.
        delta_smooth_unshrink: Share of what the diffusion removed that is handed
            back per recovery pass, from 0 to 1. Zero reproduces plain diffusion.
        layer_stiffness: Overrides the payload's authored stiffness for this call.
        under_layers: The wearables equipped below this one, innermost first.
        over_layers: The wearables equipped above this one; their inner
            surfaces are the ceilings this wearable stays under.

    Returns:
        The deformed mesh and the evaluated joint positions and rotations.
    """
    start_time = time.time()
    stiffness = layer_stiffness_of({"anchorMapping": anchor_mapping, "layerStiffness": layer_stiffness})
    bind_heights = np.maximum(_gap_heights(anchor_mapping), 0.0)
    pressed = np.zeros(len(bind_heights), dtype=bool)

    # 1. The layers below, written into the template
    bare_mesh = refit_deformed_mesh
    if under_layers:
        refit_deformed_mesh = patch_refit_mesh(refit_deformed_mesh, under_layers, stiffness, over_layers)

    # 2. The gaps, shortened along the anchor normal by what would stand out
    # above (along the normal, not by scaling: a vertex on a phantom ground has
    # to go below its anchor point, which a scale cannot do)
    if under_layers or over_layers:
        stack = [*under_layers, {"anchorMapping": anchor_mapping, "layerStiffness": stiffness}, *over_layers]
        excess = _ceiling_excess(bare_mesh, refit_deformed_mesh, stack, len(under_layers))
        if excess is not None:
            # a press under a millionth of the driver is nothing: the vertex still drapes
            excess = np.where(excess > _PRESS_RATIO * float(bare_mesh.scale), excess, 0.0)
        if excess is not None and np.any(excess > 0.0):
            pressed = excess > 0.0
            _, normals = _anchor_frames(anchor_mapping)
            gaps = np.asarray(anchor_mapping["vertexAnchors"]["gapVectors"], dtype=np.float64)
            anchor_mapping = dict(anchor_mapping)
            anchor_mapping["vertexAnchors"] = dict(anchor_mapping["vertexAnchors"])
            anchor_mapping["vertexAnchors"]["gapVectors"] = (gaps - excess[:, None] * normals).tolist()

    # 3. The deformation gradient off the bare driver (a patch's cliffs stretched
    # a jacket's 2.5 cm gap into 9 cm of travel), the surface points off the patched one
    transforms, bind_edges = _deformation_gradients(anchor_mapping, bare_mesh)
    source = np.array([f["sourceFaceId"] for f in anchor_mapping["anchorSurface"]], dtype=np.int64)
    tri_verts = refit_deformed_mesh.vertices[refit_deformed_mesh.faces[source]]

    # 4. Evaluate the wearable vertices
    new_verts, _ = _unpack_anchor_set(anchor_mapping["vertexAnchors"], transforms, tri_verts, len(wearable_mesh.vertices))

    # 5. Draping: cloth that hangs off the body is draped over it rather than following it point by
    # point -- a hem between the thighs, the back over the buttocks. Where the body pushes, the cloth
    # rests on it; between two pushes it is the smooth interpolation of both, never the valley.
    # A vertex the stack presses counts as a contact. Heights are ratios of the driver's diagonal.
    if _BRIDGE_HANGING_RATIO > 0.0 and len(anchor_mapping["smoothingGraph"]["neighborIndices"]):
        graph = anchor_mapping["smoothingGraph"]
        scale = float(bare_mesh.scale)
        contact, hanging_height = _BRIDGE_CONTACT_RATIO * scale, _BRIDGE_HANGING_RATIO * scale
        bind_verts = np.asarray(wearable_mesh.vertices, dtype=np.float64)
        hanging = np.where(pressed, 0.0, bind_heights)
        touch = np.clip((hanging_height - hanging) / (hanging_height - contact), 0.0, 1.0)
        follow = new_verts - bind_verts
        driven = np.einsum('nij,nj->ni', _blend_transforms(anchor_mapping, transforms), _anchor_frames(anchor_mapping)[1])
        driven /= np.maximum(np.linalg.norm(driven, axis=1, keepdims=True), 1e-12)
        draped = _bridge_deltas(follow, touch >= 1.0, driven, np.asarray(graph["neighborIndices"], dtype=np.int64),
                                np.asarray(graph["neighborOffsets"], dtype=np.int64), _BRIDGE_TOLERANCE_RATIO * scale)
        new_verts = bind_verts + touch[:, None] * follow + (1.0 - touch)[:, None] * draped

    # 6. Delta smoothing over the baked wearable graph, its reach set by the driver's triangles
    if delta_smooth_iterations > 0:
        graph = anchor_mapping["smoothingGraph"]
        bind_verts = np.asarray(wearable_mesh.vertices, dtype=np.float32)
        driver_edge = float(np.median(bind_edges)) if len(bind_edges) else 0.0
        deltas = _smooth_deformation_deltas(
            new_verts.astype(np.float32) - bind_verts,
            np.asarray(graph["neighborIndices"], dtype=np.int64), np.asarray(graph["neighborOffsets"], dtype=np.int64),
            delta_smooth_iterations, unshrink=delta_smooth_unshrink, positions=bind_verts,
            max_length=_DELTA_DIFFUSION_TRIANGLES * driver_edge)
        new_verts = bind_verts + deltas

    deformed_wearable = wearable_mesh.copy()
    deformed_wearable.vertices = new_verts

    # 7. Evaluate the joints, with identity bind rotations when none are given
    joint_positions, joint_rotations = None, None
    num_joints = len(anchor_mapping["jointAnchors"]["anchorOffsets"]) - 1
    if num_joints > 0:
        bind_rotations = joint_bind_rotations
        if bind_rotations is None:
            bind_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], num_joints, axis=0)
        joint_positions, joint_rotations = _unpack_anchor_set(
            anchor_mapping["jointAnchors"], transforms, tri_verts, num_joints, bind_rotations=bind_rotations)

    logger.info(f"Finished apply_anchor_mapping in {time.time() - start_time:.4f} seconds.")
    return deformed_wearable, joint_positions, joint_rotations


# =============================================================================
# APPLY HELPERS
# =============================================================================
# How far the delta field may diffuse, in driver triangles, whatever the
# iteration count: two holds a large-faced panel off a body that bulges under it
# (three put 0.86 cm of a shirt inside a chest, two put 0.30).
_DELTA_DIFFUSION_TRIANGLES = 2.0
_DELTA_RECOVERY_PASSES = 2   # Recovery passes over the smoothing residual; the curve is flat past two.
_DELTA_RECOVERY_WIDTH = 2    # How much finer the recovery sweeps run than the smoothing they undo.
# Draping, as ratios of the driver's bounding diagonal (about 3 cm and 6 cm on a 2 m body): a vertex
# closer to the body than the first height is in contact and follows it, one further than the second
# hangs and is draped. Zero disables. Lower heights drape more but change convex shapes and let
# garments interpenetrate; these keep every wearable scaling with the body and bridge the thighs.
_BRIDGE_CONTACT_RATIO = 0.015
_BRIDGE_HANGING_RATIO = 0.03
_BRIDGE_TOLERANCE_RATIO = 1e-6   # Relaxation stops when nothing moves by more than this share of the diagonal.
_PRESS_RATIO = 1e-6              # A press smaller than this share of the diagonal is dropped, so it reads the same in any unit.
_ORIENTATION_MARGIN = 0.05       # How decisively the gaps must lean one way before they settle which way is out.
_BRIDGE_SWEEPS = 400             # Cap on the relaxation; a hoodie's back converges in about 300.


def _blend_transforms(mapping: RuntimeMeshAnchor, transforms: np.ndarray) -> np.ndarray:
    """The blended deformation gradient of each wearable vertex, weights as baked, identity where unanchored."""
    a = _Anchors(mapping)
    out = np.zeros((a.num_points, 3, 3))
    if not a.empty:
        np.add.at(out, a.owner, transforms[a.face_ids] * a.weights[:, None, None])
    out[np.bincount(a.owner, minlength=a.num_points) == 0] = np.eye(3)
    return out


def _bridge_deltas(
    deltas: np.ndarray,
    fixed: np.ndarray,
    normals: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_offsets: np.ndarray,
    tolerance: float
) -> np.ndarray:
    """Drapes the free vertices over the body-following deltas: harmonic where the cloth hangs, on the body where it pushes.

    An obstacle problem, solved by projected Jacobi over the wearable graph: each sweep takes every
    free vertex to the mean of its neighbours, then back up to the body-following delta along its
    normal wherever it went below. The fixed vertices are the boundary. Where the body pushes, the
    cloth rests on it and carries its neighbours; between two pushes it is the smooth interpolation
    of both.

    Args:
        deltas: (N, 3) deformed minus bind positions, the obstacle.
        fixed: (N,) the vertices in contact with the body, kept as they are.
        normals: (N, 3) driven unit normals, the direction the obstacle holds along.
        neighbor_indices: Flat CSR neighbour list of the smoothing graph.
        neighbor_offsets: Per-vertex offsets into it, length N + 1.
        tolerance: The relaxation stops once nothing moves by more than this, in world units.
    """
    # 1. The neighbour-mean operator, and who is free to move
    counts = np.diff(neighbor_offsets)
    free = ~fixed & (counts > 0)
    if not np.any(free):
        return deltas.copy()
    averaging = sparse.csr_matrix((np.repeat(1.0 / np.maximum(counts, 1), counts), neighbor_indices, neighbor_offsets),
                                  shape=(len(deltas), len(deltas)))

    # 2. Relax the free vertices to the mean of their neighbours, never below the obstacle, until nothing moves
    field = deltas.copy()
    for _ in range(_BRIDGE_SWEEPS):
        mean = (averaging @ field)[free]
        below = np.einsum('ij,ij->i', deltas[free] - mean, normals[free])
        mean += np.maximum(below, 0.0)[:, None] * normals[free]
        change = float(np.abs(mean - field[free]).max())
        field[free] = mean
        if change < tolerance:
            break
    return field


def _smooth_deformation_deltas(
    deltas: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_offsets: np.ndarray,
    iterations: int,
    blend: float = 0.5,
    unshrink: float = 0.0,
    positions: Optional[np.ndarray] = None,
    max_length: Optional[float] = None
) -> np.ndarray:
    """Diffuses the per-vertex deformation deltas across the packed graph.

    Each sweep is a convex blend toward the neighbour mean, so the field obeys
    a maximum principle and never overshoots. With positions and max_length,
    each vertex's step is capped so a group of sweeps diffuses no further than
    that length whatever its edge measures -- or a coarse panel is flattened
    into the body under it. The volume the sweeps take from the largest deltas
    comes back by refining against the raw field with the same operator, swept
    finer:

        y = S(raw);  then k times:  y += unshrink * S(raw - y)

    whose gain per mode never exceeds 1. The C++ runtime performs the same
    gather-average loop over the same arrays.

    Args:
        deltas: (N, 3) deformed minus bind positions.
        neighbor_indices: Flat CSR neighbour list of the smoothing graph.
        neighbor_offsets: Per-vertex offsets into it, length N + 1.
        iterations: Sweeps of the smoothing group.
        blend: Weight of the neighbour mean per sweep.
        unshrink: Share of the residual handed back per recovery pass, 0 to 1.
        positions: Bind positions, for the length cap.
        max_length: How far a sweep group may diffuse, in world units.
    """
    # 1. The row-normalized neighbour-mean operator, a zero-copy CSR wrap
    num_points = len(deltas)
    counts = np.diff(neighbor_offsets)
    isolated = counts == 0
    averaging = sparse.csr_matrix((np.repeat((1.0 / np.maximum(counts, 1)).astype(np.float32), counts),
                                   neighbor_indices, neighbor_offsets), shape=(num_points, num_points))

    # 2. Per-vertex mean edge, for the length cap
    limited = positions is not None and max_length is not None and max_length > 0.0
    if limited:
        pts = np.asarray(positions, dtype=np.float32)
        sources = np.repeat(np.arange(num_points), counts)
        edge_lengths = np.zeros(num_points, dtype=np.float32)
        np.add.at(edge_lengths, sources, np.linalg.norm(pts[sources] - pts[neighbor_indices], axis=1))
        edge_lengths /= np.maximum(counts, 1)

    def sweep(field: np.ndarray, count: int) -> np.ndarray:
        # The step that keeps a group of `count` sweeps inside max_length
        step = blend
        if limited and count > 0:
            capped = max_length * max_length / (2.0 * count * np.maximum(edge_lengths, 1e-12) ** 2)
            step = np.minimum(blend, capped).astype(np.float32)[:, None]
        for _ in range(count):
            neighbor_mean = averaging @ field
            neighbor_mean[isolated] = field[isolated]
            field = (1.0 - step) * field + step * neighbor_mean
        return field

    # 3. Smooth, then hand back a share of the residual, swept finer
    smoothed = sweep(deltas, iterations)
    if unshrink > 0.0:
        width = max(1, iterations * _DELTA_RECOVERY_WIDTH)
        for _ in range(_DELTA_RECOVERY_PASSES):
            smoothed = smoothed + unshrink * sweep(deltas - smoothed, width)
    return smoothed
