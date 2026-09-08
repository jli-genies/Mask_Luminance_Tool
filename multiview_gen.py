import os
import copy
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import trimesh as tm
from PIL import Image

from genies.meshutils.geometry.registration.landmarks.multi_view_solver import MultiviewLandmarksBuilder, MultiviewLandmarkSolver, conform_view_token
from genies.meshutils.geometry.registration.landmarks.expander import LandmarkExpander
from genies.meshutils.geometry.registration.landmarks.constants import LandmarksNaming as LN
from genies.meshutils.geometry.registration.landmarks.constants import LandmarksPresetNaming as LPN
from genies.meshutils.geometry.registration.landmarks.landmarks_utils import predict_landmarks_from_images, convert_landmarks_names
from genies.meshutils.shading.texture_segmentation import TmTextureSegmentation
from genies.meshutils.shading.texture_from_images import TmBakeTextureFromImages
from genies.meshutils.shading.texture_utils import save_numpy_as_image, load_image_as_numpy
from genies.meshutils.callers.texture_transfer import tm_texture_transfer
from genies.meshutils.geometry.rbf import rbf_deform

logger = logging.getLogger(__name__)

DEFAULT_MULTIVIEW_PARAMS = {
    # --- Base Variables ---
    "preset": LPN.HUMANOID,
    "image_extension": ".png",
    "base_view_token": "y_0",
    "main_color_mask_file": "head_composite_skin_mask.png",
    "extrapolation_mask_file": "head_extrapolation_mask.png",
    "color_composite_file": "head_composite_color.png",
    "concept_image_prefix": "concept",
    "facing_mask_prefix": "facingWeight",
    "output_image_name": "albedo_from_concept.png",

    # --- Texture Constraints ---
    "output_size": 2048,
    "background_removal_threshold": 10,
    "dilatation_factor": 2.0,
    "background_erode_iterations": 3,
    "main_color_mask_opacity": 0.5,
    "main_color_mask_blur_opacity": 1.0,
    "main_color_mask_blur_factor": 30.0,
    "facing_blur_factor": 3.0,
    "masked_blur_factor": 3.0,
    "masked_blend_dist_factor": 15.0,
    "highlight_removal_up_threshold": 160,
    "highlight_removal_low_threshold": 10,
    "highlight_removal_transition": 20.0,
    "highlight_removal_factor": 0.0,
    "mirror_result_if_sym": False,
    "save_debug_images": False,

    # --- End Point params ---
    "confidence_threshold": 0.4,
    "visibility_threshold": 0.4,
    "use_mediapipe": False,
    "mediapipe_weight": 0.0,

    # --- Solve Constraints ---
    "perp_landmarks": {
                        "x":(LN.CHIN_CENTER, LN.NOSE_CENTER),
                        "y":(f"{LN.SIDE_LEFT}_{LN.BROW}_2", f"{LN.SIDE_RIGHT}_{LN.BROW}_2")
                    },
    "anchor_landmarks": [LN.NECK_CENTER],
    "focal_length": "orthographic",
    "template_constraint_weight": 0.0,
}

def load_view_data(
    input_images_dir: str,
    prefix: str = DEFAULT_MULTIVIEW_PARAMS["concept_image_prefix"],
    ext: str = DEFAULT_MULTIVIEW_PARAMS["image_extension"],
    call_ml_predict: bool = True,
    confidence_threshold: float = DEFAULT_MULTIVIEW_PARAMS["confidence_threshold"],
    visibility_threshold: float = DEFAULT_MULTIVIEW_PARAMS["visibility_threshold"],
    use_mediapipe: bool = DEFAULT_MULTIVIEW_PARAMS["use_mediapipe"],
    mediapipe_weight: float = DEFAULT_MULTIVIEW_PARAMS["mediapipe_weight"],
    progress_callback: Optional[Callable] = None
) -> Tuple[Dict[str, Dict], List[str]]:
    """Constructs evaluation arrays from disk images and local cache files.

    Scans the specified directory for valid image matrices. It checks for a
    local 'multiview_cache.json' file to reuse previous ML results. If a view
    is missing from the cache, it triggers the machine learning endpoints.
    Translates results into structured dictionary formats for the solvers.
    Detailed logging tracks the source of the landmark data per view.

    Args:
        input_images_dir (str): Directory containing the source images.
        prefix (str): Prefix string to filter target image files.
        ext (str): File extension to filter target image files.
        call_ml_predict (bool): Flag to trigger external prediction endpoints.
        confidence_threshold (float): Minimum score to accept a predicted landmark.
        visibility_threshold (float): Minimum score to accept landmark visibility.
        progress_callback (Optional[Callable]): Function to report execution progress.

    Returns:
        Tuple[Dict[str, Dict], List[str]]: A dictionary of image metrics and a list of view tokens.

    Raises:
        RuntimeError: If authentication tokens expire during the prediction call.
    """
    import json

    # ==========================================
    # 1. INITIALIZATION & CACHE LOADING
    # ==========================================
    # Initialize the primary storage dictionary for image properties.
    view_data = {}

    # Initialize the ordered list tracking processed view identifiers.
    view_tokens = []

    # Validate input directory to prevent access exceptions during iteration.
    if not input_images_dir or not os.path.exists(input_images_dir):
        logger.warning("Input images directory is missing. Cannot build in-memory data.")
        return view_data, view_tokens

    # Construct the path to the local landmark cache file.
    cache_path = os.path.join(input_images_dir, "multiview_cache.json")
    cache_data = {}

    # Load existing cache data if the file is present on disk.
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            logger.info(f"Load View Data: Found cache with {len(cache_data)} entries.")
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}")

    # ==========================================
    # 2. GATHER & VALIDATE TARGET FILES
    # ==========================================
    # Initialize lists to store validated file names and paths.
    valid_files = []
    all_image_paths = []

    # Iterate over directory contents to filter and validate target files.
    for filename in sorted(os.listdir(input_images_dir)):
        if filename.startswith(prefix) and filename.endswith(ext):
            full_path = os.path.join(input_images_dir, filename).replace("\\", "/")

            try:
                # Execute a lazy header check to prevent corrupt files.
                with Image.open(full_path) as img:
                    img.verify()

                valid_files.append(filename)
                all_image_paths.append(full_path)
            except Exception as e:
                logger.warning(f"Skipping corrupt image '{full_path}': {e}")

    if not valid_files:
        return view_data, view_tokens

    # ==========================================
    # 3. BATCH ML PREDICTION
    # ==========================================
    # Identify images that are not present in the local cache.
    images_to_predict = []
    predict_indices = []

    # Map current inference parameters to evaluate cache validity.
    current_predict_config = {
        "confidence_threshold": confidence_threshold,
        "visibility_threshold": visibility_threshold,
        "use_mediapipe": use_mediapipe,
        "mediapipe_weight": mediapipe_weight,
    }

    for idx, filename in enumerate(valid_files):
        # Extract and immediately conform the token
        raw_token = filename[len(prefix):-len(ext)].lstrip("_")
        token = conform_view_token(raw_token)
        entry = cache_data.get(token, {})
        cache_config = entry.get("_predict_config")

        # Bypass cache if token is missing or if the inference configurations do not match.
        if (token not in cache_data) or (cache_config != current_predict_config):
            images_to_predict.append(all_image_paths[idx])
            predict_indices.append(idx)

    # Initialize the list to store machine learning coordinates.
    prediction_results = []

    if call_ml_predict and images_to_predict:
        if progress_callback:
            progress_callback(f"Extracting ML landmarks for {len(images_to_predict)} views...")

        try:
            # Execute predictions passing the unified mediapipe arguments.
            prediction_results = predict_landmarks_from_images(
                image_paths=images_to_predict,
                confidence_threshold=confidence_threshold,
                visibility_threshold=visibility_threshold,
                use_mediapipe=use_mediapipe,
                mediapipe_weight=mediapipe_weight
            )
        except Exception as e:
            # Raise a specific error if authentication has expired.
            error_msg = str(e).lower()
            if any(k in error_msg for k in ["token", "credentials", "grant"]):
                raise RuntimeError(f"AWS authentication failed: {e}") from e
            logger.error(f"Batch ML prediction failed: {e}")

    # Map the returned prediction results back to their original indices.
    all_results = [None] * len(valid_files)
    for i, original_idx in enumerate(predict_indices):
        if i < len(prediction_results):
            all_results[original_idx] = prediction_results[i]

    # ==========================================
    # 4. PROCESS RESULTS
    # ==========================================
    for idx, filename in enumerate(valid_files):
        # Extract and immediately conform the token
        raw_token = filename[len(prefix):-len(ext)].lstrip("_")
        token = conform_view_token(raw_token)
        img_path = all_image_paths[idx]

        # Extract raster boundaries to map normalized points into pixel space.
        try:
            src_img = load_image_as_numpy(img_path)
            h, w = src_img.shape[:2]
        except Exception as e:
            logger.warning(f"Could not read image data for '{img_path}': {e}")
            continue

        view_tokens.append(token)
        names_raw, pos_px, pos_norm = [], [], []

        # Prioritize cache data over fresh machine learning results.
        if token in cache_data:
            entry = cache_data[token]
            names_raw = entry.get("names_raw", [])
            pos_px = np.array(entry.get("pos_px", []))
            pos_norm = np.array(entry.get("pos", []))
            logger.debug(f"View '{token}': Loaded {len(names_raw)} items from cache.")

        # Fallback to ML results if the cache is missing the token.
        elif all_results[idx]:
            landmark_data = all_results[idx]
            for name, norm_coords in landmark_data.items():
                names_raw.append(name)

                # The coordinate arrays intentionally use different Y-axis directions.
                # The pos_norm array uses upward-positive coordinates.
                # The downstream solver expects this mathematical format.
                # The pos_px array uses downward-positive coordinates.
                # The downstream texture baker uses this image pixel format.
                # Convert these arrays explicitly if consumer needs differ.
                nx, ny_up = norm_coords[0], 1.0 - norm_coords[1]
                pos_norm.append([nx, ny_up])
                pos_px.append([nx * w, norm_coords[1] * h])

            pos_norm = np.array(pos_norm)
            pos_px = np.array(pos_px)
            logger.debug(f"View '{token}': Loaded {len(names_raw)} items from ML.")

        # Normalize nomenclature utilizing standard pipeline conversion.
        normalized_names = convert_landmarks_names(names_raw, usd_to_maya=True)
        logger.info(f"View '{token}': Finalized {len(normalized_names)} landmarks.")

        # Populate the primary dictionary with the finalized metadata.
        view_data[token] = {
            "image_path": img_path,
            "names_raw": names_raw,
            "names": normalized_names,
            "pos_px": pos_px if len(pos_px) > 0 else np.empty((0, 2)),
            "pos": pos_norm if len(pos_norm) > 0 else np.empty((0, 2))
        }

    return view_data, view_tokens

def expand_view_data(
    view_data: Dict[str, Any],
    template_3d_landmarks: Tuple[List[str], np.ndarray],
    perp_landmarks: Optional[Dict[str, Tuple[str, str]]] = DEFAULT_MULTIVIEW_PARAMS["perp_landmarks"],
    preset: str = DEFAULT_MULTIVIEW_PARAMS["preset"],
    background_removal_threshold: int = DEFAULT_MULTIVIEW_PARAMS["background_removal_threshold"],
    cache: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Tuple[List[str], np.ndarray]], Dict[str, Any]]:
    """Extracts spatial data and builds 2D landmarks for computations.

    Checks the provided memory dictionary to bypass redundant processing.
    Delegates to the builder to construct procedural mappings on the projection.

    Args:
        view_data: Source image configurations and positions.
        template_3d_landmarks: Tuple holding names and coordinates of reference points.
        preset: Anatomical layout string for expansion rules.
        cache: Dictionary to store and retrieve processing stages.
        perp_landmarks (Optional[Tuple[str, str]]): Reference points for rotation.

    Returns:
        Tuple with triangulated target arrays and expanded 2D view mappings.
    """
    # ==========================================
    # 1. CACHE RETRIEVAL
    # ==========================================
    # Inspect dictionary array to bypass system calculations
    if cache is not None and "solver_landmarks" in cache and "expanded_view_data" in cache:
        return cache["solver_landmarks"], cache["expanded_view_data"]

    # ==========================================
    # 2. LANDMARKS EXPANSION
    # ==========================================
    # Instantiate the builder to evaluate target silhouette definitions
    builder = MultiviewLandmarksBuilder(
        view_data=copy.deepcopy(view_data),
        template_3d_landmarks=template_3d_landmarks,
        perp_landmarks=perp_landmarks,
        preset=preset,
        background_removal_threshold=background_removal_threshold
    )

    # Trigger internal solvers to calculate spatial constraint maps
    solver_landmarks = builder.run()

    # Isolate the final projected properties for subsequent texture math
    expanded_view_data = copy.deepcopy(builder.view_data)

    # ==========================================
    # 3. CACHE POPULATION
    # ==========================================
    # Register compiled metrics back into memory structure
    if cache is not None:
        cache["solver_landmarks"] = solver_landmarks
        cache["expanded_view_data"] = expanded_view_data

    return solver_landmarks, expanded_view_data

def tm_texture_from_images(
    # --- Geometry & References ---
    mesh_tm: tm.Trimesh,
    view_data: Dict[str, Any],
    template_3d_landmarks: Tuple[List[str], np.ndarray],
    view_tokens: List[str],

    # --- Folders ---
    input_images_dir: str = "",
    input_masks_dir: str = "",
    output_dir: str = "",
    output_image_name: str = DEFAULT_MULTIVIEW_PARAMS["output_image_name"],

    # --- String Keys ---
    preset: str = DEFAULT_MULTIVIEW_PARAMS["preset"],
    base_view_token: str = DEFAULT_MULTIVIEW_PARAMS["base_view_token"],

    # --- Image Integrations ---
    composite_texture_paths: Optional[Dict[str, Any]] = None,
    main_color_mask_file: Optional[str] = DEFAULT_MULTIVIEW_PARAMS.get("main_color_mask_file"),
    extrapolation_mask_file: Optional[str] = DEFAULT_MULTIVIEW_PARAMS.get("extrapolation_mask_file"),
    color_composite_file: Optional[str] = DEFAULT_MULTIVIEW_PARAMS.get("color_composite_file"),

    # --- Operations Settings ---
    output_size: int = DEFAULT_MULTIVIEW_PARAMS["output_size"],
    background_removal_threshold: int = DEFAULT_MULTIVIEW_PARAMS["background_removal_threshold"],
    background_erode_iterations: int = DEFAULT_MULTIVIEW_PARAMS["background_erode_iterations"],
    main_color_mask_opacity: float = DEFAULT_MULTIVIEW_PARAMS.get("main_color_mask_opacity", 0.25),
    main_color_mask_blur_opacity: float = DEFAULT_MULTIVIEW_PARAMS.get("main_color_mask_blur_opacity", 1.0),
    main_color_mask_blur_factor: float = DEFAULT_MULTIVIEW_PARAMS.get("main_color_mask_blur_factor", 10.0),
    facing_blur_factor: float = DEFAULT_MULTIVIEW_PARAMS["facing_blur_factor"],
    masked_blur_factor: float = DEFAULT_MULTIVIEW_PARAMS["masked_blur_factor"],
    masked_blend_dist_factor: float = DEFAULT_MULTIVIEW_PARAMS["masked_blend_dist_factor"],
    dilatation_factor: float = DEFAULT_MULTIVIEW_PARAMS["dilatation_factor"],
    highlight_removal_up_threshold: int = DEFAULT_MULTIVIEW_PARAMS["highlight_removal_up_threshold"],
    highlight_removal_low_threshold: int = DEFAULT_MULTIVIEW_PARAMS["highlight_removal_low_threshold"],
    highlight_removal_transition: float = DEFAULT_MULTIVIEW_PARAMS["highlight_removal_transition"],
    highlight_removal_factor: float = DEFAULT_MULTIVIEW_PARAMS["highlight_removal_factor"],
    mirror_result_if_sym: bool = DEFAULT_MULTIVIEW_PARAMS["mirror_result_if_sym"],
    save_debug_images: bool = DEFAULT_MULTIVIEW_PARAMS["save_debug_images"],

    # --- Advanced Segmentation Constraints ---
    segmentation_dict: Optional[Dict[str, Any]] = None,

    # --- Geometric Normal Map ---
    create_geometric_normal_map: bool = False,
    geometric_normal_map_name: str = "normal",
    normal_source_mesh_tm: Optional[tm.Trimesh] = None,

    # --- Caching ---
    cache: Optional[Dict[str, Any]] = None
) -> Union[str, np.ndarray]:
    """Projects image observations into a comprehensive UV mapped texture.

    Evaluates multiview camera projections to build a unified base texture.
    Applies spatial segmentation profiles directly on top of the generated canvas.
    Loads and saves data matrices tracking the specific generation states.

    Args:
        mesh_tm (tm.Trimesh): The source mesh geometry receiving the bake.
        view_data (Dict[str, Any]): Camera configuration and structural positions.
        template_3d_landmarks (Tuple[List[str], np.ndarray]): Spatial reference coordinates.
        view_tokens (List[str]): Identifiers mapping to active camera views.
        input_images_dir (str): Base directory path for projection sources.
        input_masks_dir (str): Base directory path for alpha constraints.
        output_dir (str): Path to store diagnostic and final outputs.
        output_image_name (str): Filename for the finalized texture array.
        preset (str): Configuration preset string mapping anatomical rules.
        base_view_token (str): Primary frontal camera identifier token.
        composite_texture_paths (Optional[Dict[str, Any]]): Overlay images for detail injection.
        main_color_mask_file=mask for compositing the skin tone on top of results.
        color_composite_file=color file for compositing on top of results.
        extrapolation_mask_file=mask for extrapolating existing colors to uncovered areas.
        output_size (int): Resolution of the target square texture matrix.
        background_removal_threshold (int): Tolerance for floodfill alpha isolation.
        background_erode_iterations (int): Intensity of the mask border erosion.
        main_color_mask_opacity (float): Overlay transparency limit for tone logic.
        main_color_mask_blur_opacity (float): Transparency limit for blurred tone matrices.
        main_color_mask_blur_factor (float): Radius scale for topological softening.
        facing_blur_factor (float): Mask smoothing scale evaluated from face angles.
        masked_blur_factor (float): Scale defining topological blend transitions.
        masked_blend_dist_factor (float): Padding scale for synthetic gap extrapolation.
        dilatation_factor (float): Absolute outward push protecting UV gutters.
        highlight_removal_up_threshold (int): Clamp limit for bright luminance extremes.
        highlight_removal_low_threshold (int): Clamp limit securing deep shadow integrity.
        highlight_removal_transition (float): Roll-off curve smoothing the clamp zones.
        highlight_removal_factor (float): Interpolation factor for the neutralized data.
        mirror_result_if_sym (bool): Enforces bilateral symmetry across the texture.
        save_debug_images (bool): Triggers persistent storage of intermediate arrays.
        segmentation_dict (Optional[Dict[str, Any]]): 3D spatial zones to mask the result.
        create_geometric_normal_map (bool): Instructs the baker to autonomously generate a procedural geometric normal map.
        geometric_normal_map_name (str): The output filename suffix designating the procedural normal map.
        normal_source_mesh_tm (Optional[tm.Trimesh]): High-resolution geometry utilized to raycast and bake the geometric normal map.
        cache (Optional[Dict[str, Any]]): Memory reference holding spatial metrics.

    Returns:
        Union[str, np.ndarray]: Absolute file path or the generated RGB/RGBA matrix.
    """
    # ==========================================
    # 1. DATA PREPARATION
    # ==========================================
    # Project reference vectors to construct standardized multiview tracking arrays.
    _, expanded_view_data = expand_view_data(
        view_data=view_data,
        template_3d_landmarks=template_3d_landmarks,
        preset=preset,
        background_removal_threshold=background_removal_threshold,
        cache=cache
    )

    # Initialize discrete mapping dictionaries routing absolute image paths.
    concept_images = {}
    concept_images_landmarks = {}

    # Map mathematical coordinate spaces explicitly to spatial source bounds.
    for token, data in expanded_view_data.items():
        img_path = data.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue

        pos_px = data["pos_px"]
        concept_images[token] = img_path
        concept_images_landmarks[token] = (data["names"], np.array(pos_px))

    # Extract standard nomenclature arrays to unify spatial identifier keys.
    clean_lm_names, t_pos = template_3d_landmarks

    # Convert string identities sequentially to support structural namespace matching.
    clean_lm_names = [n.split(':')[-1] for n in clean_lm_names]
    clean_lm_names = convert_landmarks_names(clean_lm_names, usd_to_maya=True)

    # ==========================================
    # 2. TEXTURE GENERATION
    # ==========================================
    composite_data = {}
    if composite_texture_paths:
        diffuse_color = composite_texture_paths.get("diffuseColor")
        if diffuse_color is not None:
            composites = []
            # Normalize the input to a list to handle single string paths safely.
            diffuse_items = diffuse_color if isinstance(diffuse_color, (list, tuple)) else [diffuse_color]
            for p in diffuse_items:
                if not p:
                    continue
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing composite texture: {p}")
                composites.append(load_image_as_numpy(p))
            if composites:
                composite_data["diffuseColor"] = composites

    # Extract the extrapolation mask from the dictionary mapping
    baker = TmBakeTextureFromImages(
        mesh_tm=mesh_tm,
        landmarks_3d_names=clean_lm_names,
        landmarks_3d_pos=t_pos,
        concept_images_landmarks=concept_images_landmarks,
        concept_images=concept_images,
        concept_images_dir=input_images_dir,
        input_masks_dir=input_masks_dir,
        output_dir=output_dir,
        output_image_name=output_image_name,
        view_tokens=view_tokens,
        base_view_token=base_view_token,
        composite_textures=composite_data,
        main_color_mask_file=main_color_mask_file,
        extrapolation_mask_file=extrapolation_mask_file,
        color_composite_file=color_composite_file,
        output_image_size=output_size,
        background_removal_threshold=background_removal_threshold,
        main_color_mask_opacity=main_color_mask_opacity,
        main_color_mask_blur_opacity=main_color_mask_blur_opacity,
        main_color_mask_blur_factor=main_color_mask_blur_factor,
        facing_blur_factor=facing_blur_factor,
        masked_blur_factor=masked_blur_factor,
        masked_blend_dist_factor=masked_blend_dist_factor,
        dilatation_factor=dilatation_factor,
        background_erode_iterations=background_erode_iterations,
        highlight_removal_up_threshold=highlight_removal_up_threshold,
        highlight_removal_low_threshold=highlight_removal_low_threshold,
        highlight_removal_transition=highlight_removal_transition,
        highlight_removal_factor=highlight_removal_factor,
        mirror_result_if_sym=mirror_result_if_sym,
        save_debug_images=save_debug_images
    )

    final_result = baker.run()

    # ==========================================
    # 3. NORMAL MAP GENERATION
    # ==========================================
    # If a high-res source mesh is provided, trigger the geometric transfer pipeline
    # to exclusively bake the procedural normal map.
    if create_geometric_normal_map and normal_source_mesh_tm is not None:
        tgt_tex = {}
        tgt_tex[geometric_normal_map_name] = os.path.join(output_dir, f"head_concept_{geometric_normal_map_name}.png").replace("\\", "/")

        # Isolate the normal map composite texture from the general dictionary
        normal_composites = None
        # Extract texture targets.
        if composite_texture_paths and geometric_normal_map_name in composite_texture_paths:
            norm_val = composite_texture_paths[geometric_normal_map_name]

            # Normalize sequence inputs to standard lists to guarantee path validation.
            normal_items = norm_val if isinstance(norm_val, (list, tuple)) else [norm_val]
            normal_composites = {geometric_normal_map_name: list(normal_items)}

        # Delegate to the texture transfer utility to bake raycasted deltas.
        tm_texture_transfer(
            src_mesh=normal_source_mesh_tm,
            tgt_mesh=mesh_tm,
            src_texture_paths={},
            tgt_texture_paths=tgt_tex,
            create_geometric_normal_map=True,
            use_height_as_normal=False,
            tgt_composite_texture_paths=normal_composites,
            # We omit the segmentation dictionary to preserve the eyebrows shape.
            segmentation_dict=None,
            normal_match_threshold=0.0
        )

    # ==========================================
    # 4. SEGMENTATION MASKING POST-PROCESS
    # ==========================================
    if not segmentation_dict:
        return final_result

    # Generate the segmentation masks tracking spatial mesh partitions.
    segment_transfer_mask, segment_remove_mask = None, None
    if segmentation_dict:
        segmenter = TmTextureSegmentation(
            mesh_tm=mesh_tm,
            output_size=output_size,
        )
        segment_transfer_mask, segment_remove_mask, _ = segmenter.generate_masks(segmentation_dict)

    # Evaluate variable types mapping matrix arrays or string destinations.
    is_path = isinstance(final_result, str)
    if is_path:
        tex_array = load_image_as_numpy(final_result)
        base, ext = os.path.splitext(final_result)
        final_fmt = ext.lower().lstrip('.') or "png"
    else:
        tex_array = final_result

    # Identify physical data maps preventing unintended color modifications.
    is_data_map = any(k in output_image_name.lower() for k in ["roughness", "metallic", "metalness", "orm", "ambientocclusion"])

    # Cache unsegmented assets allowing diagnostic pipeline reviews.
    if is_path:
        raw_path = f"{base}_unsegmented{ext}"
        save_numpy_as_image(tex_array, raw_path, fmt=final_fmt)

    # The segmenter alpha behavior serves as a final safety boundary.
    tex_array = segmenter.apply_to_texture(
        tex_array=tex_array,
        transfer_mask=segment_transfer_mask,
        remove_mask=segment_remove_mask,
        strip_alpha_and_fill=is_data_map,
        raster_data=baker.raster_data,
        mesh_data=baker.mesh_data
    )

    # Store finished arrays replacing generic diagnostic versions.
    if is_path:
        save_numpy_as_image(tex_array, final_result, fmt=final_fmt)


    if is_path:
        return final_result

    return tex_array

def tm_mesh_from_images(
    # --- Targets ---
    mesh_tm: tm.Trimesh,
    view_data: Dict[str, Any],
    template_3d_landmarks: Tuple[List[str], np.ndarray],

    # --- Optics & Adjustments ---
    preset: str = DEFAULT_MULTIVIEW_PARAMS["preset"],
    background_removal_threshold: int = DEFAULT_MULTIVIEW_PARAMS["background_removal_threshold"],
    focal_length: Union[str, float] = DEFAULT_MULTIVIEW_PARAMS["focal_length"],
    base_view_token: str = DEFAULT_MULTIVIEW_PARAMS["base_view_token"],
    template_constraint_weight: float = DEFAULT_MULTIVIEW_PARAMS["template_constraint_weight"],
    perp_landmarks: Optional[Dict[str, Tuple[str, str]]] = DEFAULT_MULTIVIEW_PARAMS["perp_landmarks"],
    anchor_landmarks: Optional[List[str]] = DEFAULT_MULTIVIEW_PARAMS["anchor_landmarks"],

    # --- Modifiers ---
    base_skeleton: Optional[Any] = None,
    mirror_if_sym: bool = True,
    cache: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generates a deformed 3D mesh by matching triangulated landmarks against an expanded 3D template.

    Args:
        mesh_tm (tm.Trimesh): The source mesh geometry to deform.
        view_data (Dict[str, Any]): The multi-view configuration and coordinates.
        template_3d_landmarks (Tuple[List[str], np.ndarray]): Reference 3D spatial dataset.
        preset (str, optional): Configuration preset for expansion. Defaults to DEFAULT_MULTIVIEW_PARAMS["preset"].
        background_removal_threshold (int, optional): Threshold for silhouette extraction. Defaults to DEFAULT_MULTIVIEW_PARAMS["background_removal_threshold"].
        focal_length (Union[str, float], optional): Camera lens depth parameter. Defaults to DEFAULT_MULTIVIEW_PARAMS["focal_length"].
        base_view_token (str, optional): Primary structural reference token. Defaults to DEFAULT_MULTIVIEW_PARAMS["base_view_token"].
        template_constraint_weight (float, optional): Influence factor for template depth. Defaults to DEFAULT_MULTIVIEW_PARAMS["template_constraint_weight"].
        perp_landmarks (Optional[Dict[str, Tuple[str, str]]], optional): Reference points defining rotation axes. Defaults to DEFAULT_MULTIVIEW_PARAMS["perp_landmarks"].
        anchor_landmarks (Optional[List[str]], optional): Landmarks mapped strictly to the template. Defaults to DEFAULT_MULTIVIEW_PARAMS["anchor_landmarks"].
        base_skeleton (Optional[Any], optional): Skeleton object for kinematic lookups. Defaults to None.
        mirror_if_sym (bool, optional): Toggle for bilateral symmetry enforcement. Defaults to True.
        cache (Optional[Dict[str, Any]], optional): Dictionary to store calculation states. Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]: The deformed vertex array and triangulated landmark positions.
    """
    # ==========================================
    # 1. 3D TEMPLATE EXPANSION
    # ==========================================
    # Extract names and coordinate arrays from the provided template.
    clean_lm_names, t_pos = template_3d_landmarks

    # Convert string identifiers to match standard maya conventions.
    maya_names = convert_landmarks_names(clean_lm_names, usd_to_maya=True)
    template_dict = dict(zip(maya_names, t_pos, strict=True))

    # Retrieve the appropriate preset rules specific to two-dimensional workflows.
    preset_2d = f"{preset}_2d" if f"{preset}_2d" in LPN.ALL_PRESETS else preset

    # Instantiate the three-dimensional expander module.
    src_expander = LandmarkExpander(
        base_skeleton=base_skeleton,
        mesh_tm=mesh_tm,
        landmarks_pos=np.array(list(template_dict.values())),
        landmarks_names=list(template_dict.keys()),
        preset=preset_2d
    )

    # Generate auxiliary geometric landmarks across the mesh surface.
    src_exp_pos, src_exp_names = src_expander.run()

    # Package the expanded elements into a unified structure.
    expanded_template_3d_landmarks = (src_exp_names, src_exp_pos)

    # ==========================================
    # 2. 2D DATA PREPARATION
    # ==========================================
    # Project two-dimensional landmarks analyzing silhouette bounds.
    solver_landmarks, _ = expand_view_data(
        view_data=view_data,
        template_3d_landmarks=expanded_template_3d_landmarks,
        perp_landmarks=perp_landmarks,
        preset=preset,
        background_removal_threshold=background_removal_threshold,
        cache=cache
    )

    # ==========================================
    # 3. TRIANGULATION COMPUTATION
    # ==========================================
    # Instantiate the multi-view solver logic.
    solver = MultiviewLandmarkSolver(
        input_images_landmarks=solver_landmarks,
        perp_landmarks=perp_landmarks,
        anchor_landmarks=anchor_landmarks,
        template_3d_landmarks=expanded_template_3d_landmarks,
        template_constraint_weight=template_constraint_weight,
        focal_length=focal_length,
        base_view_token=base_view_token,
        mirror=mirror_if_sym
    )

    # Calculate unified spatial structures extrapolating hidden points.
    calculated_3d = solver.run()

    # ==========================================
    # 4. PURGE IGNORED LANDMARKS
    # ==========================================
    # Isolate solved elements missing from the expanded source map.
    keys_to_remove = [k for k in list(calculated_3d.keys()) if k not in src_exp_names]

    # Remove invalid targets to ensure exact dimensional parity.
    for k in keys_to_remove:
        del calculated_3d[k]

    final_src_pos = []
    final_tgt_pos = []

    # Populate arrays aligning corresponding source and target pairs.
    for name, s_pos in zip(src_exp_names, src_exp_pos, strict=True):
        if name in calculated_3d:
            final_src_pos.append(s_pos)
            final_tgt_pos.append(calculated_3d[name])

    # ==========================================
    # 5. DEFORMATION SOLVE
    # ==========================================
    # Bypass execution if spatial targets are mathematically insufficient.
    if len(final_src_pos) < 4:
        logger.warning(
            f"Insufficient landmark matches ({len(final_src_pos)}) for RBF deformation. "
            "Returning original mesh vertices."
        )
        return mesh_tm.vertices, calculated_3d

    # Apply multidimensional interpolation shifting original vertices.
    deformed_verts = rbf_deform(
        points_to_deform=mesh_tm.vertices,
        src_landmarks_pos=np.array(final_src_pos),
        tgt_landmarks_pos=np.array(final_tgt_pos),
        kernel="linear",
        degree=0,
        smoothing=0.0
    )

    return deformed_verts, calculated_3d