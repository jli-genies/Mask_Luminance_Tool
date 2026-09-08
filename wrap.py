import os
import trimesh as tm
import numpy as np
import logging
from typing import Dict, Optional

from genies.meshutils.geometry.utils import tm_copy
from genies.meshutils.geometry.registration.register import TmRegister

logger = logging.getLogger(__name__)

DEFAULT_REGISTRATION_PARAMS = {
    # --- Global Settings & Pre-processing ---
    "random_seed": 42,
    "rbf_kernel": "linear",
    "mirror_result_if_sym": True,
    "iterations": 15,
    "bidirectional_correspondence": True,

    # --- Bounding Box Alignment (used if no landmarks) ---
    "bbox_align_if_no_landmarks": False,
    "bbox_use_centroid": False,
    "bbox_stretch_factor": 0.5,

    # --- Stiffness Weights ---
    "stiffness_weight": 0.5,
    "stiffness_start_end_weights": (100.0, 1.0),
    "stiffness_iter_decay": 2.0,

    # --- Landmark Constraint Weights ---
    "landmarks_weight": 1.0,
    "landmarks_start_end_weights": (1.0, 1.0),
    "landmarks_rigidity_radius_factor": 0.035,

    # --- Rigid Vertex Constraints ---
    "src_rigid_vertices_weight": 1.0,
    "src_rigid_vertices_dropoff_dist_factor": 0.025,
    "src_rigid_vertices_thickness_factor": 0.01,
    "src_rigid_vertices_boundary_neighb": 3,

    # --- Excluded Target Vertices ---
    "tgt_island_size_thresh": 0.35,
    "tgt_island_noise_thresh": 0.25,

    # --- Correspondence Search Feature Weights ---
    "sim_max_neighbors": 20,
    "sim_dist_weight": 1.0,
    "sim_norm_weight": 1.0,
    "sim_curv_weight": 0.25,
    "sim_thick_weight": 0.25,

    # --- Correspondence Search Tuning ---
    "sim_min_score": 0.5,
    "sim_max_closest_dist_factor": 0.025,
    "sim_search_dist_factor": 0.5,
    "sim_min_search_dist_factor": 0.01,
    "sim_max_search_dist_factor": 0.02,

    # --- Surface Projection Step ---
    "project_weight": 0.75,
    "project_iter_decay": 2.0,
    "project_normal_threshold": 0.25,

    # --- Post-Processing ---
    "deltamush_weight": 0.0,
    "volume_preservation_weight": 1.0,
    "volume_exaggeration": 0.0
}

def tm_mesh_wrap(
    source_mesh: tm.Trimesh,
    target_mesh: tm.Trimesh,
    source_landmarks: Optional[np.ndarray] = None,
    target_landmarks: Optional[np.ndarray] = None,
    weight_maps_dir: str = None,
    **kwargs
):
    """
    Wraps the source mesh to the target mesh using MeshMatch registration.

    This function first checks if the source and target meshes share the same topology
    (identical vertex counts and face connectivity). If they do, it returns a copy
    of the target mesh immediately, skipping expensive registration.

    If a `weight_maps_dir` is provided, it parses the directory for texture maps
    to control spatial variations of registration parameters.

    Args:
        source_mesh (tm.Trimesh): The source mesh to be deformed.
        target_mesh (tm.Trimesh): The target mesh to wrap onto.
        source_landmarks (np.ndarray): Landmark coordinates on the source mesh.
        target_landmarks (np.ndarray): Corresponding landmark coordinates on the target mesh.
        weight_maps_dir (str, optional): Path to a folder containing weight map textures.
        **kwargs: See the docstring of `tm_mesh_wrap_meshmatch` for available options.

    Returns:
        tm.Trimesh: The wrapped mesh (or a copy of the target mesh if topology matches).
    """
    # --- 1. Topology Check ---
    # Check if meshes have identical topology (same number of vertices/faces and identical face indices).
    # If so, the source is already compatible with the target; we return the target geometry directly.
    is_same_shape = (
        source_mesh.vertices.shape == target_mesh.vertices.shape and
        source_mesh.faces.shape == target_mesh.faces.shape
    )

    if is_same_shape and np.array_equal(source_mesh.faces, target_mesh.faces):
        logger.info("Topology match detected. Returning target mesh copy.")
        return tm_copy(target_mesh)

    # --- 2. Parse Weight Maps ---
    # We parse the folder once here and pass the dictionary to the methods.
    textures_weights = parse_weight_maps_folder(weight_maps_dir, prefix="register")

    # --- 3. Register ---
    return tm_mesh_wrap_meshmatch(
        source_mesh,
        target_mesh,
        source_landmarks,
        target_landmarks,
        textures_weights=textures_weights,
        **kwargs
    )

def parse_weight_maps_folder(folder_path: str,
                             prefix: str = "register") -> Dict[str, str]:
    """
    Parses a directory for image files to build a parameter-to-filepath mapping.

    This helper iterates through a folder, finding valid image files.
    It automatically strips the prefix from the filename to match the
    internal parameter names of TmRegister.

    Args:
        folder_path (str): The path to the directory containing weight maps.
        prefix (str): A prefix to filter weight map files.

    Returns:
        Dict[str, str]: A dictionary where keys are parameter names (PREFIX STRIPPED)
                        and values are absolute file paths.
    """
    textures_weights = {}

    if not folder_path or not os.path.isdir(folder_path):
        if folder_path:
            logger.warning(f"Weight maps folder not found: {folder_path}")
        return textures_weights

    # Supported image extensions
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)

        # Check extension, ignore hidden files, check prefix
        if ext.lower() in valid_extensions and not filename.startswith('.') and name.startswith(prefix):
            full_path = os.path.abspath(os.path.join(folder_path, filename))

            # Strip the prefix so the key matches the internal parameter name ---
            # e.g., "register_stiffness_weight" -> "stiffness_weight"
            key_name = name[len(prefix):]

            # Handle accidental leading underscores if user passed prefix="register"
            # but file is "register_stiffness.png" (resulting in "_stiffness")
            if key_name.startswith("_"):
                key_name = key_name[1:]

            if key_name:
                textures_weights[key_name] = full_path
                logger.info(f"Found weight map for parameter '{key_name}': {full_path}")

    return textures_weights

def tm_mesh_wrap_meshmatch(
    source_mesh: tm.Trimesh,
    target_mesh: tm.Trimesh,
    source_landmarks: Optional[np.ndarray] = None,
    target_landmarks: Optional[np.ndarray] = None,
    textures_weights: Optional[Dict[str, str]] = None,
    **kwargs
):
    """
    Wraps the source mesh using the custom TmRegister class.

    This function passes all keyword arguments and the parsed texture weights
    directly to the `TmRegister.set_parameters` method.

    Args:
        source_mesh (tm.Trimesh): The source mesh.
        target_mesh (tm.Trimesh): The target mesh.
        source_landmarks (np.ndarray): The source landmarks.
        target_landmarks (np.ndarray): The target landmarks.
        textures_weights (Dict[str, str]): Dictionary mapping parameter names to image paths.
        **kwargs: See the docstring for `TmRegister.set_parameters` for all
                  available configuration options.
    """
    # 1. Initialize the registration class.
    use_outer_shell_mesh = kwargs.get("use_outer_shell", False)
    register = TmRegister(src_mesh_tm=source_mesh,
                          tgt_mesh_tm=target_mesh,
                          use_outer_shell_mesh=use_outer_shell_mesh)

    # 2. Merge defaults with provided kwargs to ensure consistent parameters.
    merged_kwargs = DEFAULT_REGISTRATION_PARAMS.copy()
    merged_kwargs.update(kwargs)

    # 3. Set all parameters, passing landmarks and other kwargs through.
    register.set_parameters(
        textures_weights=textures_weights,
        src_landmarks_pos=source_landmarks,
        tgt_landmarks_pos=target_landmarks,
        **merged_kwargs
    )

    # 4. Run the registration process.
    register.run()

    # 4. Return the resulting deformed mesh
    return register.out_mesh_tm