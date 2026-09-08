import os
import logging
from typing import Dict, Union, Tuple, Optional, List, Any
import trimesh as tm
import numpy as np
import cv2

from genies.meshutils.shading.texture_baker import TmTextureBaker
from genies.meshutils.shading.texture_segmentation import TmTextureSegmentation
from genies.meshutils.shading.texture_utils import load_image_as_numpy, save_numpy_as_image, resize_image, generate_normal_from_height

logger = logging.getLogger(__name__)

DEFAULT_TEXTURE_PARAMS = {
    # --- Resolution & Sampling ---
    "resolution": "auto",
    "max_resolution": 2048,
    "precision_scale": 1.0,

    # --- Projection & Mapping Logic ---
    "max_search_dist_factor": 0.05,
    "normal_match_threshold": 0.5,
    "search_candidates": 8,

    # --- Post-Processing & Filtering ---
    "dilation_scale": 0.005,
    "extrapolation_scale": 0.05,
    "use_volumetric_extrapolation": True,
    "filter_type": "gaussian",
    "post_filter_size": 0.5,

    # --- Compositing & Color (UI specific handling) ---
    "brightness": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "exposure": 0.0,
    "gamma": 1.0,
    "assign_new_material": True,

    # --- Geometric Normal Generation ---
    "use_height_as_normal": False,
    "create_geometric_normal_map": True,
    "geometric_normal_map_name": "normal",

    # --- IO & Serialization ---
    "file_format": "auto",

    # --- Advanced Segmentation Constraints ---
    "segmentation_dict": None,
}

def tm_texture_transfer(
                    # --- Inputs ---
                    src_mesh: tm.Trimesh,
                    tgt_mesh: tm.Trimesh,
                    src_texture_paths: Dict[str, str],
                    tgt_texture_paths: Dict[str, str],
                    tgt_composite_texture_paths: Optional[Dict[str, List[str]]] = None,
                    tgt_blur_file: Optional[str] = None,
                    tgt_extrapolation_file: Optional[Union[str, List[str]]] = None,

                    # --- Resolution & Sampling ---
                    resolution: Union[str, Tuple[int, int], float, int] = DEFAULT_TEXTURE_PARAMS["resolution"],
                    max_resolution: Optional[int] = DEFAULT_TEXTURE_PARAMS["max_resolution"],
                    precision_scale: float = DEFAULT_TEXTURE_PARAMS["precision_scale"],

                    # --- Projection & Mapping Logic ---
                    max_search_dist_factor: float = DEFAULT_TEXTURE_PARAMS["max_search_dist_factor"],
                    normal_match_threshold: float = DEFAULT_TEXTURE_PARAMS["normal_match_threshold"],
                    search_candidates: int = DEFAULT_TEXTURE_PARAMS["search_candidates"],

                    # --- Post-Processing & Filtering ---
                    dilation_scale: float = DEFAULT_TEXTURE_PARAMS["dilation_scale"],
                    extrapolation_scale: float = DEFAULT_TEXTURE_PARAMS["extrapolation_scale"],
                    use_volumetric_extrapolation: bool = DEFAULT_TEXTURE_PARAMS["use_volumetric_extrapolation"],
                    filter_type: str = DEFAULT_TEXTURE_PARAMS["filter_type"],
                    post_filter_size: float = DEFAULT_TEXTURE_PARAMS["post_filter_size"],

                    # --- Compositing & Color ---
                    color_settings: Optional[Dict[str, float]] = None,

                    # --- Geometric Normal Generation ---
                    use_height_as_normal: bool = DEFAULT_TEXTURE_PARAMS["use_height_as_normal"],
                    create_geometric_normal_map: bool = DEFAULT_TEXTURE_PARAMS["create_geometric_normal_map"],
                    geometric_normal_map_name: str = DEFAULT_TEXTURE_PARAMS["geometric_normal_map_name"],
                    geometric_normals_smooth_factor: float = 4.0,

                    # --- IO & Serialization ---
                    file_format: str = DEFAULT_TEXTURE_PARAMS["file_format"],

                    # --- Advanced Segmentation Constraints ---
                    segmentation_dict: Optional[Dict[str, Any]] = None,

                ) -> Dict[str, str]:
    """Transfers texture data from a high-res source mesh to a target mesh.

    Executes the primary baking pipeline. Evaluates 3D coordinates to project source
    surface data into target 2D space. Implements isolation masks to remove or
    transfer designated anatomical features based on the segmentation dictionary.

    Args:
        src_mesh (tm.Trimesh): The detailed source mesh acting as the data origin.
        tgt_mesh (tm.Trimesh): The target mesh receiving the baked textures.
        src_texture_paths (Dict[str, str]): Source image paths mapped by channel.
        tgt_texture_paths (Dict[str, str]): Destination paths for the baked images.
        tgt_composite_texture_paths (Optional[Dict[str, List[str]]]): Extra textures to layer.
        tgt_blur_file (Optional[str]): Grayscale mask to drive localized smoothing.
        tgt_extrapolation_file (Optional[Union[str, List[str]]]): Grayscale mask(s)
            to synthesize gaps from. A single path, or a list of paths merged
            via pixel-wise maximum (e.g. a base mask plus a chin-area mask).
        resolution (Union[str, Tuple[int, int], float, int]): Output dimension strategy.
        max_resolution (Optional[int]): Hard limit for the maximum dimension.
        precision_scale (float): Scales the internal barycentric calculation grid.
        max_search_dist_factor (float): Projection distance relative to mesh size.
        normal_match_threshold (float): Minimum dot product required for surface alignment.
        search_candidates (int): Neighboring vertices to evaluate during projection.
        dilation_scale (float): Outward padding ratio into uninhabited UV gutters.
        extrapolation_scale (float): Boundary soften radius for synthesized regions.
        use_volumetric_extrapolation (bool): Forces 3D Laplacian diffusion for gap filling.
        filter_type (str): Convolution kernel type for global filtering.
        post_filter_size (float): Global blur radius for projection noise reduction.
        color_settings (Optional[Dict[str, float]]): Tonal adjustment settings. Accepts
            'brightness', 'contrast', 'saturation' (multipliers, 1.0 neutral),
            'exposure' (stops, 0.0 neutral), and 'gamma' (power curve, 1.0 neutral).
        use_height_as_normal (bool): Instructs the baker to generate normals from height data.
        create_geometric_normal_map (bool): Instructs the baker to generate procedural normals.
        geometric_normal_map_name (str): The output key for the procedural normal map.
        file_format (str): The format extension for saved arrays.
        segmentation_dict (Optional[Dict[str, Any]]): Defines spatial zones using 3D points.
            Accepts a "mode" parameter set to "remove" or "transfer", and an optional
            "skip_channels" list naming the output maps a zone must not touch. A zone
            applies to every channel by default, the procedural normal map included.

    Returns:
        Dict[str, str]: A dictionary mapping processed channels to output paths.
    """
    results = {}

    # ==========================================
    # 1. BAKER INITIALIZATION
    # ==========================================
    # Guard against zero or negative values causing invalid smoothing radii.
    if geometric_normals_smooth_factor <= 0:
        raise ValueError("geometric_normals_smooth_factor must be > 0")

    # Initialize the projection utility linking source and target geometries.
    baker = TmTextureBaker(
        src_mesh=src_mesh,
        tgt_mesh=tgt_mesh,
        max_search_dist_factor=max_search_dist_factor,
    )

    # Resolve reference image dimensions to establish the processing grid scale.
    ref_img = None
    if src_texture_paths:
        ref_key = next((k for k in ['baseColor', 'diffuse', 'color'] if k in src_texture_paths), None)
        if not ref_key:
            ref_key = list(src_texture_paths.keys())[0]

        p = src_texture_paths[ref_key]
        if os.path.exists(p):
            ref_img = load_image_as_numpy(p)

    # Evaluate target UV pixels against source surface coordinates.
    baker.precompute_mapping(
        resolution=resolution,
        max_resolution=max_resolution,
        precision_scale=precision_scale,
        ref_texture=ref_img,
        normal_match_threshold=normal_match_threshold,
        search_candidates=search_candidates
    )

    # Extract standard integer resolution width from the precomputed mapping metrics.
    if baker._cached_mapping is not None:
        out_res = baker._cached_mapping.get("width", 2048)
    else:
        # Fall back to standard defaults if the mapping configuration is missing.
        out_res = max_resolution if resolution == "auto" else (resolution[0] if isinstance(resolution, tuple) else 2048)

    # ==========================================
    # 2. BASE MAPS & SEGMENTATION MASKS
    # ==========================================
    # Load standard structural extrapolation maps from disk files. Accepts a
    # single path or a list of paths -- when several are given (e.g. a base
    # mask plus a chin-area mask), they're merged via pixel-wise maximum.
    extrapolation_map = None
    extrap_files = (
        [tgt_extrapolation_file] if isinstance(tgt_extrapolation_file, str)
        else list(tgt_extrapolation_file or [])
    )
    for extrap_file in extrap_files:
        if not extrap_file or not os.path.exists(extrap_file):
            continue
        try:
            raw_extrap = load_image_as_numpy(extrap_file)
            resized_extrap = resize_image(raw_extrap, out_res, out_res)
            if resized_extrap.ndim == 3:
                layer = np.mean(resized_extrap[..., :3], axis=2).astype(np.uint8)
            else:
                layer = resized_extrap.astype(np.uint8)
            extrapolation_map = layer if extrapolation_map is None else np.maximum(extrapolation_map, layer)
        except Exception as e:
            logger.warning(f"Failed to load/normalize Extrapolation Map '{extrap_file}': {e}")

    # Load standard structural blur maps from disk files.
    blur_map = None
    if tgt_blur_file and os.path.exists(tgt_blur_file):
        try:
            raw_blur = load_image_as_numpy(tgt_blur_file)
            resized_blur = resize_image(raw_blur, out_res, out_res)
            if resized_blur.ndim == 3:
                blur_map = np.mean(resized_blur[..., :3], axis=2).astype(np.uint8)
            else:
                blur_map = resized_blur.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Failed to load/normalize Blur Map: {e}")

    # Track the active region name to append to result keys for accessory isolation.
    active_transfer_region = None
    segmenter = None
    transfer_mask, remove_mask = None, None

    # Instantiate the independent segmentation class if settings are active.
    if segmentation_dict:
        segmenter = TmTextureSegmentation(
            mesh_tm=tgt_mesh,
            output_size=out_res,
        )
        transfer_mask, remove_mask, active_transfer_region = segmenter.generate_masks(segmentation_dict)

    # ==========================================
    # 3. CHANNEL BAKING (SAMPLING PHASE)
    # ==========================================
    for channel, src_path in src_texture_paths.items():

        if not os.path.exists(src_path):
            continue

        try:
            # Validate standard destination paths and ensure target directories exist.
            out_path = tgt_texture_paths.get(channel)
            if not out_path:
                continue

            out_dir = os.path.dirname(out_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Ingest image arrays strictly for downstream mathematical interpolation.
            src_img = load_image_as_numpy(src_path)

            # Evaluate required alpha composites and format them into consistent channel layers.
            tgt_composites = None
            if tgt_composite_texture_paths and channel in tgt_composite_texture_paths:
                tgt_composites = []
                for p in tgt_composite_texture_paths[channel]:
                    if p and os.path.isfile(p):
                        tgt_composites.append(load_image_as_numpy(p))

            # Enable spatial vector constraints for normal maps specifically.
            is_normal_channel = "normal" in channel.lower()

            # Flag data maps to strip the alpha channel and crush the background to black
            is_data_map = any(k in channel.lower() for k in ["roughness", "metallic", "metalness", "orm", "ambientocclusion"])

            # Resolve this channel's masks up front. The height-derived branch
            # below reads the transfer mask as its relief, so it has to see the
            # same channel exclusions as the compositing further down.
            ch_transfer_mask, ch_remove_mask = transfer_mask, remove_mask
            if segmentation_dict and segmenter is not None:
                ch_transfer_mask, ch_remove_mask, _ = segmenter.generate_masks(
                    segmentation_dict, channel=channel)

            # Execute normal map generation from mask height if requested.
            if is_normal_channel and use_height_as_normal and ch_transfer_mask is not None:
                logger.info("TEXTURE: Generating channel normal map from mask height.")
                baked_array = generate_normal_from_height(ch_transfer_mask, strength=2.0)
            else:
                # Standard projection-based bake running agnostic of the segmentation layout.
                baked_array = baker.bake_channel(
                    src_image=src_img,
                    tgt_composite_textures=tgt_composites,
                    blur_map=blur_map,
                    extrapolation_map=extrapolation_map,
                    bake_mask=None,
                    dilation_scale=dilation_scale,
                    extrapolation_scale=extrapolation_scale,
                    use_volumetric_extrapolation=use_volumetric_extrapolation,
                    filter_type=filter_type,
                    post_filter_size=post_filter_size*5,
                    color_settings=color_settings,
                    is_normal_map=is_normal_channel,
                    strip_alpha_and_fill=is_data_map
                )

            # Apply the combined inclusion and exclusion profiles on top of the results.
            base, ext = os.path.splitext(out_path)
            final_fmt = file_format if file_format != "auto" else (ext.lower().lstrip('.') or "png")

            # Dump the unmodified texture array to disk before the mask applies.
            raw_out_path = f"{base}_unsegmented{ext}"
            save_numpy_as_image(baked_array, raw_out_path, final_fmt)

            # Apply the combined inclusion and exclusion profiles on top of the
            # results, using the masks this channel was allowed to see.
            if segmentation_dict and segmenter is not None:
                baked_array = segmenter.apply_to_texture(
                    tex_array=baked_array,
                    transfer_mask=ch_transfer_mask,
                    remove_mask=ch_remove_mask,
                    strip_alpha_and_fill=is_data_map
                )

            # Write the final segmented image matrix back to the storage drive.
            save_numpy_as_image(baked_array, out_path, final_fmt)

            res_key = f"{channel}_{active_transfer_region}" if active_transfer_region else channel
            results[res_key] = out_path

        except Exception as e:
            logger.error(f"  [Failed] Error baking {channel}: {e}", exc_info=True)

    # ==========================================
    # 4. PROCEDURAL NORMAL MAP GENERATION
    # ==========================================
    # Resolve what the normal map is allowed to see before deciding whether to
    # generate it: a region that opted out of this map must not drive its relief.
    geo_transfer_mask, geo_remove_mask = transfer_mask, remove_mask
    if segmentation_dict and segmenter is not None:
        geo_transfer_mask, geo_remove_mask, _ = segmenter.generate_masks(
            segmentation_dict, channel=geometric_normal_map_name)

    if create_geometric_normal_map or (use_height_as_normal and geo_transfer_mask is not None):
        out_path = tgt_texture_paths.get(geometric_normal_map_name)

        if out_path:
            try:
                # Evaluate height map logic independently of 3D topological generation
                if use_height_as_normal and geo_transfer_mask is not None:
                    logger.info(f"TEXTURE: Generating normal map from mask height for {active_transfer_region}.")
                    geo_normal_array = generate_normal_from_height(geo_transfer_mask, strength=2.0)
                else:
                    # Standard projection evaluating absolute orientation against source meshes.
                    # Merge spatial components strictly through normal-aware vector blending operations.
                    geo_composites = None
                    if tgt_composite_texture_paths:
                        geo_composites = []
                        for k, paths in tgt_composite_texture_paths.items():
                            if "normal" in k.lower() or k == geometric_normal_map_name:
                                for p in paths:
                                    if p and os.path.isfile(p):
                                        geo_composites.append(load_image_as_numpy(p))

                        if not geo_composites:
                            geo_composites = None

                    # Generate the raw mathematical topological normal map buffer.
                    geo_normal_array = baker.bake_geometric_normals(
                        dilation_scale=dilation_scale,
                        post_filter_size=post_filter_size*geometric_normals_smooth_factor,
                        blur_map=blur_map,
                        tgt_composite_textures=geo_composites,
                        bake_mask=None
                    )

                # Apply the combined inclusion and exclusion profiles post-generation.
                base, ext = os.path.splitext(out_path)
                final_fmt = file_format if file_format != "auto" else (ext.lower().lstrip('.') or "png")

                # Dump the unmodified normal array to disk before the mask applies.
                raw_out_path = f"{base}_unsegmented{ext}"
                save_numpy_as_image(geo_normal_array, raw_out_path, final_fmt)

                # Apply the combined inclusion and exclusion profiles post-generation,
                # using the masks this map was allowed to see.
                if segmentation_dict and segmenter is not None:
                    geo_normal_array = segmenter.apply_to_texture(
                        tex_array=geo_normal_array,
                        transfer_mask=geo_transfer_mask,
                        remove_mask=geo_remove_mask,
                        strip_alpha_and_fill=False
                    )

                # Write the final segmented normal matrix back to the storage drive.
                save_numpy_as_image(geo_normal_array, out_path, final_fmt)

                # Use the same suffix logic for geometric normal keys.
                res_key = f"{geometric_normal_map_name}_{active_transfer_region}" if active_transfer_region else geometric_normal_map_name
                results[res_key] = out_path

            except Exception as e:
                logger.error(f"  [Failed] Error generating Geometric Normal Map: {e}", exc_info=True)

    return results