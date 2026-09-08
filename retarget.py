import logging
from typing import Dict, Optional, Any, Tuple, Union

import numpy as np
import trimesh as tm

from genies.meshutils.geometry.mesh_deform.mesh_retarget import TmMeshRetarget
from genies.meshutils.geometry.mesh_deform.proximity_wrap import TmProximityWrap
from genies.meshutils.geometry.mesh_deform.refit import TmRefit

logger = logging.getLogger(__name__)

DEFAULT_RETARGET_PARAMS = {
    "local_adapt_weight": 1.0,
    "smooth_weight": 0.8,
    "stabilization_strength": 0.002,
    "normal_match_threshold": 0.85,
    "max_search_dist_factor": 0.025,
    "max_influences": 20,
    "scale_k_ring": 2,
    "scale_smooth_alpha": 100.0
}

# Global array for structural eyelid gaps > This belongs to GAP CORE BUT I AM NOT SURE WHERE TO PUT IT ATM
EYELID_PAIRS = np.array([
    # Left Eye
    [954, 976], [953, 951], [979, 982], [988, 1601], [1627, 991],
    [960, 950], [965, 962], [985, 994], [986, 995], [964, 963],
    [961, 956], [1626, 992], [989, 1600], [980, 981], [958, 957],
    [959, 975],
    # Right Eye
    [74, 96], [73, 71], [99, 102], [108, 741], [770, 111],
    [80, 70], [85, 82], [105, 114], [106, 115], [84, 83],
    [81, 76], [769, 112], [109, 740], [100, 101], [78, 77],
    [79, 95]
], dtype=np.int64)


def tm_mesh_retarget(
                    # Input meshes
                    src_bind_tm: tm.Trimesh,
                    src_registered_bind_tm: tm.Trimesh,
                    src_deformed_tm_dict: Dict[str, tm.Trimesh],
                    to_wrap_bind_tm_dict: Optional[Dict[str, tm.Trimesh]] = None,

                    # Retargeting options
                    to_wrap_methods: Optional[Dict[str, str]] = None,
                    local_adapt_weight: float = DEFAULT_RETARGET_PARAMS["local_adapt_weight"],
                    smooth_weight: float = DEFAULT_RETARGET_PARAMS["smooth_weight"],
                    stabilization_strength: float = DEFAULT_RETARGET_PARAMS["stabilization_strength"],

                    # Mapping search options
                    normal_match_threshold: float = DEFAULT_RETARGET_PARAMS["normal_match_threshold"],
                    max_search_dist_factor: float = DEFAULT_RETARGET_PARAMS["max_search_dist_factor"],
                    max_influences: int = DEFAULT_RETARGET_PARAMS["max_influences"],
                    scale_k_ring: int = DEFAULT_RETARGET_PARAMS["scale_k_ring"],
                    scale_smooth_alpha: float = DEFAULT_RETARGET_PARAMS["scale_smooth_alpha"],
                    gap_fill_pairs: Optional[np.ndarray] = None,

                    # Index Filters
                    src_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
                    tgt_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
                    tgt_excluded_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,

                    progress_callback: Optional[any] = None
                ) -> Dict[str, Dict[str, tm.Trimesh]]:
    """
    Batch processes mesh retargeting for a set of source deformations across multiple target meshes.

    This function initializes a primary retargeter (TmMeshRetarget) for the registered mesh
    and uses TmRefit for each secondary 'wrapped' mesh. It then iterates through all provided
    source deformations, applying the deformation transfer to all targets simultaneously.

    Args:
        src_bind_tm (trimesh.Trimesh): The original source mesh in its rest/bind pose.
        src_registered_bind_tm (trimesh.Trimesh): The source mesh already registered/wrapped to
            the target character's rest pose. Used as the primary correspondence bridge.
        src_deformed_tm_dict (Dict[str, trimesh.Trimesh]): A dictionary of source deformations.
        to_wrap_bind_tm_dict (Optional[Dict[str, trimesh.Trimesh]]): A dictionary of additional target
            meshes (accessories, clothes) that need to follow the source deformations.
        to_wrap_methods (Optional[Dict[str, str]]): Dictionary specifying the refit_type ('bary' or 'rbf')
            for each wrapped mesh by name. Defaults to 'bary'.
        local_adapt_weight (float): Blends between World-Space Deltas (0.0) and Surface-Space Deltas (1.0).
        smooth_weight (float): Regularization for the final Position Solve (0.1 = Sharp, 1.0 = Smooth).
        stabilization_strength (float): Factor to stabilize the rotation calculation (default 0.002).
        normal_match_threshold (float): Threshold for normal alignment during mapping for all meshes.
        max_search_dist_factor (float): Search radius for surface projection for all meshes.
        max_influences (int): Maximum number of source entities (faces for bary, vertices for RBF)
            to bind to each target vertex.
        scale_k_ring (int): Topological expansion ring count for local scale precomputation.
        scale_smooth_alpha (float): Regularization weight for the local scale field Laplacian smoothing.
        gap_fill_pairs (Optional[np.ndarray]): Explicit vertex pairs to calculate structural scale.
        src_indices: Optional array or dict of indices on the Source (Provider) to use.
        tgt_indices: Optional array or dict of indices on the Target (Receiver) to deform.
        tgt_excluded_indices: Optional array or dict of indices on the Target (Receiver) to exclude.
        progress_callback: Optional callback function to report progress.

    Returns:
        Dict[str, Dict[str, trimesh.Trimesh]]: A nested dictionary structured as:
            { "deformation_name": { "main": trimesh_result, "to_wrap_name": trimesh_result, ... } }
    """
    # ---------------------------------------------------------
    # 1. INITIALIZATION PHASE
    # ---------------------------------------------------------
    # Set up the retargeting solvers and configuration once.
    # This bakes all the mapping data (barycentric/RBF graphs) offline before the loop.
    solvers_data = initialize_retargeting_solvers(
        src_bind_tm=src_bind_tm,
        src_registered_bind_tm=src_registered_bind_tm,
        to_wrap_bind_tm_dict=to_wrap_bind_tm_dict,
        to_wrap_methods=to_wrap_methods,
        normal_match_threshold=normal_match_threshold,
        max_search_dist_factor=max_search_dist_factor,
        max_influences=max_influences,
        scale_k_ring=scale_k_ring,
        scale_smooth_alpha=scale_smooth_alpha,
        gap_fill_pairs=gap_fill_pairs,
        src_indices=src_indices,
        tgt_indices=tgt_indices,
        tgt_excluded_indices=tgt_excluded_indices
    )

    # Dictionary to hold the final processed meshes
    results = {}

    # Extract the pre-initialized solver data for easy access during the loop
    main_data = solvers_data["main"]
    to_wrap_data_dict = solvers_data.get("wrapped", {})

    logger.info(f"Processing batch of {len(src_deformed_tm_dict)} deformations.")

    # ---------------------------------------------------------
    # 2. BATCH EXECUTION PHASE
    # ---------------------------------------------------------
    # Iterate over every deformation provided in the input dictionary.
    for def_name, def_tm in src_deformed_tm_dict.items():
        if progress_callback:
            progress_callback(f"Solving: {def_name}")
        logger.info(f"Processing deformation: {def_name}")

        # Initialize the nested dictionary for this specific deformation
        results[def_name] = {}

        # ---------------------------------------------------------
        # A. Solve Main Registered Mesh (ARAP Solver)
        # ---------------------------------------------------------
        # Extract the solver and wrapper dedicated to the main character body/face
        main_solver = main_data["solver"]
        main_wrap = main_data["wrapper"]

        # Sample the deformed source vertices onto the target topology using proximity mapping
        main_sampled = main_wrap.sample(def_tm.vertices)

        # Run the ARAP (As-Rigid-As-Possible) solver to regularize the deformation
        main_result = main_solver.run(
            src_deformed_sampled=main_sampled,
            local_adapt_weight=local_adapt_weight,
            smooth_weight=smooth_weight,
            stabilization_strength=stabilization_strength,
        )

        # Store the solved main mesh in the results dictionary
        results[def_name]["main"] = main_result

        # ---------------------------------------------------------
        # B. Solve Wrapped Meshes (TmRefit)
        # ---------------------------------------------------------
        # Iterate over all additional meshes (accessories, clothes, etc.)
        for to_wrap_name, to_wrap_data in to_wrap_data_dict.items():
            # Extract the dedicated TmRefit instance for this accessory
            refitter = to_wrap_data["refitter"]

            # Apply the deformation directly using the pre-baked mapping.
            # We explicitly pass the registered bind mesh to isolate morphological
            # scale and prevent animation stretch (like blinks) from inflating accessories.
            to_wrap_result, _ = refitter.apply(
                src_wrapped_tm=def_tm,
                src_scale_ref_mesh=src_registered_bind_tm
            )

            # Store the successfully deformed accessory in the results dictionary
            results[def_name][to_wrap_name] = to_wrap_result

    logger.info("Batch retargeting complete.")
    return results

def initialize_retargeting_solvers(
                    src_bind_tm: tm.Trimesh,
                    src_registered_bind_tm: tm.Trimesh,
                    to_wrap_bind_tm_dict: Optional[Dict[str, tm.Trimesh]] = None,
                    to_wrap_methods: Optional[Dict[str, str]] = None,
                    normal_match_threshold: float = 0.85,
                    max_search_dist_factor: float = 0.025,
                    max_influences: int = 20,
                    scale_k_ring: int = 1,
                    scale_smooth_alpha: float = 2000.0,
                    gap_fill_pairs: Optional[np.ndarray] = None,
                    src_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
                    tgt_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
                    tgt_excluded_indices: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
                ) -> Dict[str, Any]:
    """
    Initializes and returns the retargeting solvers and wrappers.

    This function sets up TmMeshRetarget for the main character mesh.
    It also sets up TmRefit instances for any secondary wrapped meshes.

    Args:
        src_bind_tm (trimesh.Trimesh): The original source mesh in its rest pose.
        src_registered_bind_tm (trimesh.Trimesh): The source mesh registered to the target.
        to_wrap_bind_tm_dict (Optional[Dict[str, trimesh.Trimesh]]): Accessory meshes to deform.
        to_wrap_methods (Optional[Dict[str, str]]): Refit types ('bary' or 'rbf') per wrapped mesh.
        normal_match_threshold (float): Threshold for normal alignment during mapping.
        max_search_dist_factor (float): Search radius factor for surface projection.
        max_influences (int): Maximum source entities to bind to each target vertex.
        scale_k_ring (int): Topological expansion ring count for local scale precomputation.
        scale_smooth_alpha (float): Regularization weight for the local scale field.
        gap_fill_pairs (Optional[np.ndarray]): Explicit vertex pairs to calculate structural scale.
        src_indices (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Index mask for the Source.
        tgt_indices (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Index mask for the Target.
        tgt_excluded_indices (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Exclusion mask for the Target.

    Returns:
        Dict[str, Any]: A structured dictionary containing initialized solvers and refitters.
    """
    # Ensure dictionaries exist to prevent runtime lookup errors.
    to_wrap_bind_tm_dict = to_wrap_bind_tm_dict or {}
    to_wrap_methods = to_wrap_methods or {}

    # Master dictionary to store instantiated solver objects.
    solvers_data = {"wrapped": {}}

    def get_param(param, key):
        """Extracts a specific parameter from a dictionary or returns the global parameter."""
        if isinstance(param, dict):
            return param.get(key, None)
        return param

    def _validate_indices(mesh_tm, indices, *, name: str):
        """Ensures provided index arrays fall within the actual vertex limits.

        Args:
            mesh_tm (trimesh.Trimesh): The mesh object providing the vertex count.
            indices (np.ndarray | list | None): The target indices to validate.
            name (str): The variable name used for error reporting.

        Returns:
            np.ndarray | None: A flattened 1D array of valid integer indices.

        Raises:
            ValueError: When any index falls outside the valid geometry limits.
        """
        # Return nothing when the input is null.
        if indices is None:
            return None

        # Convert the input to a flat integer array.
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)

        # Return nothing when the array is empty.
        if indices.size == 0:
            return None

        # Count the available vertices on the mesh.
        n_verts = len(mesh_tm.vertices)

        # Filter the array for out-of-bound values.
        invalid = indices[(indices < 0) | (indices >= n_verts)]

        # Raise an exception when invalid entries exist.
        if invalid.size:
            raise ValueError(
                f"{name} contains out-of-range vertex indices for mesh with {n_verts} vertices: "
                f"{invalid[:10].tolist()}"
            )

        # Return the clean index array.
        return indices

    # ---------------------------------------------------------
    # 1. SETUP MAIN RETARGETER
    # ---------------------------------------------------------
    logger.info("Initializing Main Retargeter and Wrapper...")

    # Extract and validate indices for the source and target.
    main_src_idx = _validate_indices(src_bind_tm, get_param(src_indices, "target"), name="src_indices['target']")
    main_tgt_idx = get_param(tgt_indices, "target")
    main_exc_idx = get_param(tgt_excluded_indices, "target")

    # Validate the target indices against the registered bind mesh geometry.
    main_tgt_indices_valid = _validate_indices(src_registered_bind_tm, main_tgt_idx, name="tgt_indices['target']")
    main_tgt_excluded_valid = _validate_indices(src_registered_bind_tm, main_exc_idx, name="tgt_excluded_indices['target']")

    # Create the proximity wrapper to handle geometric sampling.
    main_wrap = TmProximityWrap(
        src_mesh_bind=src_bind_tm,
        tgt_mesh_bind=src_registered_bind_tm,
        src_mesh_registered=src_registered_bind_tm,
        normal_match_threshold=normal_match_threshold,
        max_search_dist_factor=max_search_dist_factor,
        src_indices=main_src_idx,
        tgt_indices=main_tgt_indices_valid,
        tgt_excluded_indices=main_tgt_excluded_valid
    )

    # Pre-calculate the sampled source bind geometry mapped to the target topology.
    main_src_bind_sampled = main_wrap.sample(src_bind_tm.vertices)

    # Instantiate the main deformation solver and pass the structural gap pairs.
    main_solver = TmMeshRetarget(
        tgt_bind=src_registered_bind_tm,
        src_bind_sampled=main_src_bind_sampled,
        scale_k_ring=scale_k_ring,
        scale_smooth_alpha=scale_smooth_alpha,
        gap_fill_pairs=gap_fill_pairs
    )

    # Store the main objects in the master dictionary for batch execution.
    solvers_data["main"] = {
        "solver": main_solver,
        "wrapper": main_wrap
    }

    # ---------------------------------------------------------
    # 2. SETUP WRAPPED RETARGETERS
    # ---------------------------------------------------------
    if to_wrap_bind_tm_dict:
        logger.info(f"Initializing {len(to_wrap_bind_tm_dict)} Wrapped Retargeters...")

        # Iterate through every secondary mesh defined in the dictionary.
        for name, to_wrap_tm in to_wrap_bind_tm_dict.items():
            # Retrieve the specific wrap method or fallback to barycentric interpolation.
            refit_type = to_wrap_methods.get(name, "bary")

            # Validate indices for the wrapped components.
            wrap_src_idx = _validate_indices(src_bind_tm, get_param(src_indices, name), name=f"src_indices[{name!r}]")
            wrap_tgt_idx = _validate_indices(to_wrap_tm, get_param(tgt_indices, name), name=f"tgt_indices[{name!r}]")
            wrap_exc_idx = _validate_indices(to_wrap_tm, get_param(tgt_excluded_indices, name), name=f"tgt_excluded_indices[{name!r}]")

            # Instantiate TmRefit to handle positional tracking without ARAP overhead.
            refitter = TmRefit(
                src_base_tm=src_bind_tm,
                tgt_base_tm=to_wrap_tm,
                refit_type=refit_type,
                normal_match_threshold=normal_match_threshold,
                max_search_dist_factor=max_search_dist_factor,
                max_influences=max_influences,
                src_indices=wrap_src_idx,
                tgt_indices=wrap_tgt_idx,
                tgt_excluded_indices=wrap_exc_idx
            )

            # Pre-compute the spatial search trees and interpolation graphs offline.
            refitter.bake()

            # Store the ready-to-use refitter instance under its corresponding name.
            solvers_data["wrapped"][name] = {
                "refitter": refitter
            }

    return solvers_data