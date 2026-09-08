import time
import logging
from typing import List, Optional, Union, Tuple
import trimesh
import numpy as np

from genies.meshutils.geometry.mesh_deform.refit import TmRefit

logger = logging.getLogger(__name__)

DEFAULT_REFIT_PARAMS = {
    "refit_type": "bary",
    "auto_scale": True,
    "align_rigidly": None,
    "align_vtx_id": None,
    "align_scale": (1.0, 1.0, 1.0),
    "normal_match_threshold": -1.0,
    "max_search_dist_factor": 1.0,
    "max_influences": 20,
    "mirror_if_sym": False,
    "thickness_factor": 1.0
}

DEFAULT_RIGID_REFIT_PARAMS = {
    "refit_type": "bary",
    "align_rigidly": True,
    "align_scale": (1.0, 1.0, 1.0),
    "align_vtx_id": None,
    "normal_match_threshold": -1.0,
    "max_search_dist_factor": 1.0,
    "max_influences": 100
}

DEFAULT_SOFT_REFIT_PARAMS = {
    "refit_type": "bary",
    "normal_match_threshold": -1.0,
    "max_search_dist_factor": 1.0,
    "max_influences": 5,
}

DEFAULT_EXTRAP_REFIT_PARAMS = {
    "refit_type": "rbf",
    "normal_match_threshold": -1.0,
    "max_search_dist_factor": 0.2,
    "max_influences": 20,
}

def tm_refit(src_base_tm: Union[trimesh.Trimesh, List[trimesh.Trimesh]],
             src_wrapped_tm: Union[trimesh.Trimesh, List[trimesh.Trimesh]],
             tgt_base_tms: List[trimesh.Trimesh],
             src_scale_ref_tms: Optional[Union[trimesh.Trimesh, List[trimesh.Trimesh]]] = None,
             refit_type: str = DEFAULT_REFIT_PARAMS["refit_type"],
             auto_scale: bool = DEFAULT_REFIT_PARAMS["auto_scale"],
             align_rigidly: Optional[List[bool]] = DEFAULT_REFIT_PARAMS["align_rigidly"],
             align_vtx_id: Optional[List[int]] = DEFAULT_REFIT_PARAMS["align_vtx_id"],
             align_from_pos: Optional[np.ndarray] = None,
             align_scale: Tuple[float, float, float] = DEFAULT_REFIT_PARAMS["align_scale"],
             normal_match_threshold: float = DEFAULT_REFIT_PARAMS["normal_match_threshold"],
             max_search_dist_factor: float = DEFAULT_REFIT_PARAMS["max_search_dist_factor"],
             max_influences: int = DEFAULT_REFIT_PARAMS["max_influences"],
             mirror_if_sym: bool = DEFAULT_REFIT_PARAMS["mirror_if_sym"],
             thickness_factor: float = DEFAULT_REFIT_PARAMS["thickness_factor"],
             src_indices: Optional[List[Optional[np.ndarray]]] = None) -> List[Tuple[trimesh.Trimesh, Optional[np.ndarray]]]:
    """
    Refits target meshes to match the source wrapped mesh using the TmRefit pipeline.

    This function prototypes the Bake & Lookup architecture by instantiating a TmRefit
    object for each target mesh, baking the mapping data offline, and applying the
    deformation. Optionally applies scale compensation, global matrix offsets, or
    rigidly aligns a target mesh based on the computed translation offset of a specific vertex.

    Args:
        src_base_tm (Union[trimesh.Trimesh, List[trimesh.Trimesh]]): The original source base mesh,
            or a list of meshes to be combined and treated as a single source mesh.
        src_wrapped_tm (Union[trimesh.Trimesh, List[trimesh.Trimesh]]): The deformed/wrapped source mesh,
            or a list of meshes corresponding to the source base meshes.
        tgt_base_tms (List[trimesh.Trimesh]): A list of target base meshes to refit.
        refit_type (str): The deformation method to use. Options are "bary" for barycentric
            coordinates or "rbf" for Radial Basis Function interpolation.
        auto_scale (bool): Whether to apply volume scaling (gap expansion)
            based on the source mesh's local edge length changes. Primarily applies to "bary".
        align_rigidly (Optional[List[bool]]): A list indicating whether to perform a rigid
            alignment instead of full deformation for each target mesh. Must match the
            length of tgt_base_tms.
        align_vtx_id (Optional[List[int]]): A list of vertex IDs used to calculate the
            rigid translation offset for each target mesh. Must match the length of
            tgt_base_tms.
        align_from_pos (Optional[np.ndarray]): Explicit world coordinates (N, 3) for
            direct positional alignment. Bypasses soft deformation if provided.
        align_scale (Tuple[float, float, float]): A tuple representing the scale factors
            (x, y, z) to apply during rigid alignment.
        normal_match_threshold (float): Minimum dot product between target
            normal and source normal to consider a point 'valid' (bary only). Range [-1.0, 1.0].
        max_search_dist_factor (float): Multiplier for mesh size to determine maximum
            search radius for validity checking and geodesic extrapolation.
        max_influences (int): Maximum number of source entities (faces for bary,
            vertices for RBF) to bind to each target vertex.
        mirror_if_sym (bool): If True, and the target mesh is symmetrical, enforces
            symmetry on the final vertex positions.
        thickness_factor (float): The minimum distance factor scale to maintain between the
            target vertices and the source mesh when resolving penetrations.
        src_indices (Optional[List[Optional[np.ndarray]]]): A list of specific source mesh
            indices for calculation. Must match the length of tgt_base_tms.

    Returns:
        List[Tuple[trimesh.Trimesh, Optional[np.ndarray]]]: A list of tuples containing the
            newly created deformed/aligned target mesh and its corresponding 4x4 transform
            matrix if rigidly aligned (otherwise None).

    Raises:
        ValueError: If align_rigidly, align_vtx_id, or src_indices lengths do not match tgt_base_tms.
    """
    start_time = time.time()
    logger.info(f"--- Starting Refit Pipeline (Type: {refit_type}) ---")

    # ---------------------------------------------------------
    # 1. VALIDATION
    # ---------------------------------------------------------
    num_targets = len(tgt_base_tms)

    if align_rigidly is not None and len(align_rigidly) != num_targets:
        raise ValueError("Length of align_rigidly must match length of tgt_base_tms.")

    if align_vtx_id is not None and len(align_vtx_id) != num_targets:
        raise ValueError("Length of align_vtx_id must match length of tgt_base_tms.")

    if src_indices is not None and len(src_indices) != num_targets:
        raise ValueError("Length of src_indices must match length of tgt_base_tms.")

    created_meshes = []
    total_verts = 0

    # ---------------------------------------------------------
    # 2. PROCESS TARGETS
    # ---------------------------------------------------------
    for i, tgt_base_tm in enumerate(tgt_base_tms):
        target_start = time.time()
        tgt_count = len(tgt_base_tm.vertices)
        total_verts += tgt_count

        try:
            current_src_indices = src_indices[i] if src_indices is not None else None
            if current_src_indices is not None and len(current_src_indices) > 0:
                logger.info(f"Processing target {i}: from source vertices {current_src_indices}")

            # A. Setup Refit Pipeline
            refitter = TmRefit(
                src_base_tm=src_base_tm,
                tgt_base_tm=tgt_base_tm,
                refit_type=refit_type,
                normal_match_threshold=normal_match_threshold,
                max_search_dist_factor=max_search_dist_factor,
                max_influences=max_influences,
                src_indices=current_src_indices
            )

            # B. Bake Mapping Data (Offline Simulation)
            refitter.bake()

            # C. Apply Deformation (Runtime Simulation)
            is_rigid = align_rigidly[i] if align_rigidly is not None else False
            vtx_id = align_vtx_id[i] if align_vtx_id is not None else None
            from_pos = align_from_pos[i] if align_from_pos is not None else None

            # Restrict penetration prevention to soft refits only
            created_mesh, transform_matrix = refitter.apply(
                src_wrapped_tm=src_wrapped_tm,
                src_scale_ref_mesh=src_scale_ref_tms,
                auto_scale=auto_scale,
                align_rigidly=is_rigid,
                align_vtx_id=vtx_id,
                align_scale=align_scale,
                align_from_pos=from_pos,
                mirror_if_sym=mirror_if_sym,
                thickness_factor=thickness_factor
            )

            created_meshes.append((created_mesh, transform_matrix))

            # Log & Store Stats
            dt = time.time() - target_start
            logger.info(f"Processed target {i}: {tgt_count} verts in {dt:.3f}s")

        except Exception as e:
            logger.error(f"Failed to refit target {i}: {e}", exc_info=True)

    # ---------------------------------------------------------
    # 3. FINAL REPORT
    # ---------------------------------------------------------
    total_time = time.time() - start_time
    logger.info(f"--- Refit Complete: {total_verts} verts in {total_time:.3f}s ---")

    return created_meshes