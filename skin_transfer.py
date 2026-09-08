import logging
from typing import Dict, Optional, Union

import numpy as np
import trimesh as tm

from genies.meshutils.skin.skin_data import TmSkinData
from genies.meshutils.skin.skin_transfer import TmSkinTransfer

from genies.meshutils.skin.skin_transfer import TmSkinTransfer

logger = logging.getLogger(__name__)

DEFAULT_SKIN_PARAMS = {
    # --- Post-Process Options ---
    "max_influences": 4,
    "prune_threshold": 0.02,
    "normalize_weights": True,
    "smooth_iterations": 3,
    "max_joint_dist_factor": 0.0,
    "side_purge_strictness": 0.25,
    "side_purge_weight": 1.0,

    # --- Wrapper Options ---
    "normal_match_threshold": -1.0,
    "max_search_dist_factor": 0.0,
    "max_vtx_influences": 2
}

def tm_skin_transfer(
    # --- Input Data ---
    src_mesh: tm.Trimesh,
    src_skin_data: TmSkinData,
    tgt_mesh_dict: Dict[str, tm.Trimesh],

    # --- Transfer Options ---
    joint_mapping: Optional[Dict[str, str]] = None,

    # --- Post-Process Options ---
    max_influences: int = DEFAULT_SKIN_PARAMS["max_influences"],
    prune_threshold: float = DEFAULT_SKIN_PARAMS["prune_threshold"],
    normalize_weights: bool = DEFAULT_SKIN_PARAMS["normalize_weights"],
    smooth_iterations: int = DEFAULT_SKIN_PARAMS["smooth_iterations"],
    max_joint_dist_factor: float = DEFAULT_SKIN_PARAMS["max_joint_dist_factor"],
    side_purge_strictness: float = DEFAULT_SKIN_PARAMS["side_purge_strictness"],
    side_purge_weight: float = DEFAULT_SKIN_PARAMS["side_purge_weight"],

    # --- Wrapper Options ---
    normal_match_threshold: Union[float, Dict[str, float]] = DEFAULT_SKIN_PARAMS["normal_match_threshold"],
    max_search_dist_factor: Union[float, Dict[str, float]] = DEFAULT_SKIN_PARAMS["max_search_dist_factor"],
    max_vtx_influences: Union[int, Dict[str, int]] = DEFAULT_SKIN_PARAMS["max_vtx_influences"],

    # --- Wrapper Index Filters ---
    # Can be a single array (applied to all) or a dict {tgt_name: array}
    src_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    tgt_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    tgt_excluded_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
) -> Dict[str, TmSkinData]:
    """
    Transfers skin weights from a Source mesh to one or more Target meshes using
    topology-independent Proximity Wrapping.

    This is a batch-processing caller that delegates the actual geometric
    transfer to the TmSkinTransfer class.

    Args:
        src_mesh (tm.Trimesh): The source mesh geometry. Must match vertex count of src_skin_data.
        src_skin_data (TmSkinData): The skin weights and joint names for the source.
        tgt_mesh_dict (Dict[str, tm.Trimesh]): Dictionary of target meshes mapped by name.
        joint_mapping (Optional[Dict[str, str]]): Optional dictionary to rename joints during transfer.
        max_influences (int): Maximum number of joints allowed per vertex.
        prune_threshold (float): Minimum weight threshold; values below this are zeroed.
        normalize_weights (bool): Whether to re-normalize weights to 1.0 after transfer and processing.
        smooth_iterations (int): Number of topological smoothing passes applied to the target weights.
        max_joint_dist_factor (float): Multiplier against the mesh size to determine the
            spatial filtering radius for joints. If > 0.0, weights from physically distant joints are suppressed.
        normal_match_threshold (Union[float, Dict[str, float]]): Minimum dot product between source and
            target normals to consider a mapping valid. Can be a global float or dict per target.
        max_search_dist_factor (Union[float, Dict[str, float]]): Multiplier against the target mesh
            bounding box diagonal to set the maximum spatial search distance. Can be a global float or dict per target.
        max_vtx_influences (Union[int, Dict[str, int]]): Maximum number of source faces to blend per target
            vertex during spatial query. Can be a global int or dict per target.
        src_indices (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Source vertex indices to use.
        tgt_indices (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Target vertex indices to transfer onto.
        tgt_excluded_indices (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Target vertex indices to skip.

    Returns:
        Dict[str, TmSkinData]: The transferred and processed skin data mapped by target mesh name.
    """
    results = {}

    logger.info(f"Starting skin transfer for {len(tgt_mesh_dict)} target(s)...")

    for tgt_name, tgt_mesh in tgt_mesh_dict.items():
        logger.info(f"Processing target: {tgt_name}")

        # --- 1. Resolve Per-Target Parameters ---
        # Helper to safely extract value from scalar, array, or dictionary.
        # Differentiates between a Dict (per-target) and an Array (global input).
        def get_param(param, key, default):
            if isinstance(param, dict):
                return param.get(key, default)
            return param

        current_threshold = get_param(normal_match_threshold, tgt_name, 0.0)
        current_max_dist = get_param(max_search_dist_factor, tgt_name, 0.0)
        current_max_vtx_influences = get_param(max_vtx_influences, tgt_name, 1)

        # Resolve index masks
        current_src_idx = get_param(src_indices, tgt_name, None)
        current_tgt_idx = get_param(tgt_indices, tgt_name, None)
        current_exc_idx = get_param(tgt_excluded_indices, tgt_name, None)

        # --- 2. Initialize the Single Mesh Transfer ---
        transfer_solver = TmSkinTransfer(
            src_mesh=src_mesh,
            tgt_mesh=tgt_mesh,
            normal_match_threshold=current_threshold,
            max_search_dist_factor=current_max_dist,
            max_vtx_influences=current_max_vtx_influences,
            src_indices=current_src_idx,
            tgt_indices=current_tgt_idx,
            tgt_excluded_indices=current_exc_idx
        )

        # --- 3. Run Transfer with Smoothing and Proximity Filters ---
        results[tgt_name] = transfer_solver.run(
            src_skin_data=src_skin_data,
            joint_mapping=joint_mapping,
            max_influences=max_influences,
            prune_threshold=prune_threshold,
            normalize_weights=normalize_weights,
            smooth_iterations=smooth_iterations,
            max_joint_dist_factor=max_joint_dist_factor,
            side_purge_strictness=side_purge_strictness,
            side_purge_weight=side_purge_weight
        )

    logger.info("Skin transfer complete.")
    return results