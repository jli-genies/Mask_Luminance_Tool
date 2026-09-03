import logging
from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree

from genies.meshutils import exchange
from genies.meshutils.geometry.mesh_data.mesh_data import TmMeshData
from genies.meshutils.shading.texture_utils import (
  apply_color_correction,
  apply_extrapolation_blur,
  extrapolate_texture_volumetric,
  load_image_as_numpy,
  rasterize_mesh_uvs,
  save_numpy_as_image,
)


logger = logging.getLogger(__name__)

# Blur radius relative to the texture height
SOFTNESS_FACTOR = 0.005
# Flat 2D donor search radius relative to the texture height. Unrelated to
# extrapolate_texture_volumetric's own donor_max_dist_factor, which is a
# fraction of the mesh's 3D scale, not of the image.
DONOR_MAX_DIST_FACTOR = 0.03


def infill_from_mask(
  image: Union[str, np.ndarray],
  mask: Union[str, np.ndarray],
  softness_px: Optional[float] = None,
  donor_max_dist_px: Optional[float] = None,
  template_usd_file: Optional[str] = None,
  donor_max_dist_factor: Optional[float] = None,
  output_path: Optional[str] = None,
) -> np.ndarray:
  """Fills the masked region of an image with the colors that surround it.

  Every pixel the mask paints is barred from donating. Donors are searched
  within a bounded radius and blended by a distance-weighted average so the
  closest ones dominate, rather than copied from a single nearest donor --
  which reads as hard, Voronoi-like cell edges wherever a hole spans more
  than one surrounding color. The mask's own greyscale then drives the
  compositing weight between the fill and the original pixel at each point,
  so a feathered edge blends instead of cutting hard at a threshold. Holes
  wider than the search radius still fill, falling back to an unbounded
  search rather than being left raw. The mask floor is read off its own
  histogram, so a feathered or noisy mask needs no threshold. Both inputs
  are accepted either as arrays or as paths.

  Passing template_usd_file switches the fill from flat 2D nearest-donor
  search to the same mesh-aware volumetric extrapolation the texture baker
  uses: donors are found by walking the actual 3D surface, so two UV
  islands that sit close together in the atlas but are unrelated on the
  mesh (an ear next to a jawline, say) cannot bleed into each other the
  way a purely 2D search can. Leave it unset for a plain 2D image with no
  known UV layout.

  Args:
    image: The source image or the path to read it from.
    mask: The mask or its path, anything above its background level
      marking the region to repaint. Its own greyscale value -- not
      just whether it clears that floor -- sets how much of the fill
      is blended in at each pixel.
    softness_px: Blur radius dissolving the propagation ridge, and, when
      template_usd_file is given, the internal diffusion radius too.
      Defaults to a fraction of the texture height.
    donor_max_dist_px: Radius of the flat 2D donor search in pixels.
      Ignored when template_usd_file is given. Defaults to a fraction
      of the texture height.
    template_usd_file: Path to a USD file holding a mesh with UVs (e.g.
      .../head/humanoid/geo.usd). When given, the fill reads that
      mesh's UV layout and 3D coordinates and runs the real volumetric
      extrapolation instead of a flat 2D search. The image must be
      square: rasterizing a UV layout only supports one resolution
      for both dimensions.
    donor_max_dist_factor: Donor search radius as a fraction of the
      mesh's 3D scale, only used with template_usd_file. None keeps
      extrapolate_texture_volumetric's own current default.
    output_path: Destination file, when the result must be saved.

  Returns:
    np.ndarray: The infilled image as a uint8 array.

  Raises:
    ValueError: If template_usd_file is given and the image is not square.
  """
  # 1. Resolve both inputs, whichever side of the memory boundary they sit on.
  img = image if isinstance(image, np.ndarray) else load_image_as_numpy(image)
  raw_mask = mask if isinstance(mask, np.ndarray) else load_image_as_numpy(mask)

  # 2. Promote a flat buffer to the 3D layout the rest of this expects.
  if img.ndim == 2:
    img = img[..., np.newaxis]

  # 3. Reduce the mask to a single 8-bit plane and its continuous [0, 1] weight.
  plane = (raw_mask[..., 0] if raw_mask.ndim == 3 else raw_mask).astype(np.float32)
  if plane.max() <= 1.0:
    plane *= 255.0
  plane = plane.astype(np.uint8)
  alpha = plane.astype(np.float32) / 255.0

  # 4. Read the background level off the histogram rather than guessing a threshold.
  background = int(np.argmax(np.bincount(plane.ravel(), minlength=256)))
  holes = plane > background
  if holes.mean() > 0.5:
    logger.warning(
      "infill_from_mask: the painted region covers %.0f%% of the mask; its "
      "histogram mode may not be the background level, which would invert "
      "the fill.", holes.mean() * 100
    )
  if not np.any(holes):
    return img

  height, width = plane.shape
  blur_px = SOFTNESS_FACTOR * height if softness_px is None else softness_px

  # 5. Fill: mesh-aware when a template is given, flat 2D otherwise.
  if template_usd_file is not None:
    if height != width:
      raise ValueError(
        f"template_usd_file needs a square image, got {width}x{height}."
      )
    filled = _fill_from_template(
      img, holes, template_usd_file, blur_px, donor_max_dist_factor
    )
  else:
    radius_px = (DONOR_MAX_DIST_FACTOR * height if donor_max_dist_px is None
          else donor_max_dist_px)
    filled = _blend_nearest_donors(img.astype(np.float32), holes, radius_px)

  # 6. Composite the fill using the mask's own greyscale as the blend weight,
  # not just whether it cleared the background floor: a feathered edge blends
  # fill and original instead of cutting hard at the threshold.
  weight = alpha[..., np.newaxis]
  composited = img.astype(np.float32) * (1.0 - weight) + filled.astype(np.float32) * weight
  filled_img = np.clip(composited, 0, 255).astype(np.uint8)

  # 7. Dissolve the ridge the propagation leaves along the medial axis of a hole.
  if blur_px > 0.0:
    filled_img = apply_extrapolation_blur(filled_img, holes, blur_px)

  # 8. Commit the result to disk when a destination is requested.
  if output_path:
    save_numpy_as_image(filled_img, output_path)

  return filled_img


def _fill_from_template(
  img: np.ndarray,
  holes: np.ndarray,
  template_usd_file: str,
  blur_px: float,
  donor_max_dist_factor: Optional[float],
) -> np.ndarray:
  """Runs the real mesh-aware volumetric fill, reading UVs and 3D coordinates
  straight off template_usd_file rather than requiring a caller to already
  have that raster and mesh data on hand.

  Only the color channels (first 3) go through the mesh-aware diffusion; any
  further channel (typically alpha) is left untouched, matching how the rest
  of the shading pipeline treats color as strictly RGB.
  """
  mesh_tm = _load_template_mesh(template_usd_file)
  size = img.shape[0]

  raster_data = rasterize_mesh_uvs(mesh_tm, size)
  mesh_data = TmMeshData(mesh_tm, compute_edges=True, compute_uvs=True)
  valid_uv = raster_data["valid_uv_mask"]

  n = min(img.shape[2], 3)
  rgb = img[..., :3] if img.shape[2] >= 3 else np.repeat(img[..., :1], 3, axis=2)
  rgb_float = rgb.astype(np.float32) / 255.0

  extra_kwargs = {}
  if donor_max_dist_factor is not None:
    extra_kwargs["donor_max_dist_factor"] = donor_max_dist_factor

  diffused = extrapolate_texture_volumetric(
    base_texture=rgb_float,
    trusted_mask=~holes & valid_uv,
    holes_mask=holes & valid_uv,
    raster_data=raster_data,
    mesh_data=mesh_data,
    masked_blur_px=max(1.0, blur_px),
    masked_blend_px=max(1.0, blur_px) * 2.0,
    output_image_size=size,
    fade_inside_holes=True,
    hole_alpha=None, # this function's own compositing pass applies the mask.
    **extra_kwargs,
  )

  filled = img.copy()
  filled[..., :n] = np.clip(diffused[..., :n] * 255.0, 0, 255)
  return filled


def _load_template_mesh(usd_file: str):
  """Loads the first mesh prim in a USD file as a UV-mapped trimesh.

  Args:
    usd_file: Path to the USD file (e.g. a template's geo.usd).

  Returns:
    trimesh.Trimesh: The mesh with valid TextureVisuals.

  Raises:
    Exception: If the file holds no mesh primitive.
  """
  from pxr import Usd, UsdGeom

  stage = Usd.Stage.Open(usd_file)
  for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
      return exchange.usd_poly_to_tm_with_UVs(prim)
  raise Exception(f"No mesh primitive found in {usd_file}")


def _blend_nearest_donors(
  img: np.ndarray,
  holes: np.ndarray,
  max_dist: float,
  k: int = 8,
) -> np.ndarray:
  """Fills holes with a distance-weighted blend of the k nearest valid pixels,
  in flat 2D pixel space.

  Mirrors the donor search extrapolate_texture_volumetric runs on the 3D mesh
  surface, but over plain pixel coordinates since there is no mesh here: an
  unbounded, single-nearest-donor copy reads as hard, Voronoi-like cell edges
  wherever a hole spans more than one surrounding color. Donors are capped at
  max_dist and weighted with a Gaussian at a third of that radius, so the cut
  lands where the weight has already decayed to about 1%; holes with nothing
  inside the radius fall back to an unbounded k-nearest search rather than
  being left unfilled.

  Args:
    img: (H, W, C) float image buffer.
    holes: Boolean mask of pixels to fill.
    max_dist: Donor search radius in pixels. 0 or less means unbounded.
    k: Number of nearest donors to blend.

  Returns:
    np.ndarray: img with every holes pixel replaced by its blended donor color.
  """
  donors_yx = np.argwhere(~holes)
  if len(donors_yx) == 0:
    return img

  holes_yx = np.argwhere(holes)
  tree = cKDTree(donors_yx)
  k_eff = min(k, len(donors_yx))

  bounded = max_dist is not None and max_dist > 0.0
  distances, indices = tree.query(
    holes_yx, k=k_eff, distance_upper_bound=max_dist if bounded else np.inf
  )
  if k_eff == 1:
    distances, indices = distances[:, np.newaxis], indices[:, np.newaxis]

  # A miss comes back as an infinite distance and an out-of-range index.
  missed = ~np.isfinite(distances)
  indices = np.minimum(indices, len(donors_yx) - 1)

  # Widen the search for holes that found nothing inside the radius, so a
  # region wider than the radius still fills instead of staying raw.
  orphans = missed.all(axis=1)
  if bounded and np.any(orphans):
    far_d, far_i = tree.query(holes_yx[orphans], k=k_eff)
    if k_eff == 1:
      far_d, far_i = far_d[:, np.newaxis], far_i[:, np.newaxis]
    distances[orphans], indices[orphans] = far_d, far_i
    missed[orphans] = False

  # Weight by absolute distance so the nearest donors dominate. A third of the
  # radius, so the Gaussian has decayed to ~1% by the time the hard cut
  # applies and no donor is dropped while it still carried weight.
  distances = np.maximum(distances, 1e-6)
  if bounded:
    sigma = max(max_dist / 3.0, 1e-6)
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
  else:
    weights = np.exp(-0.5 * (distances / np.mean(distances, axis=1, keepdims=True)) ** 2)
  weights[missed] = 0.0

  # Past a few sigma the whole row underflows; let the nearest donor carry it
  # rather than dividing by zero.
  degenerate = weights.sum(axis=1) <= 1e-12
  if np.any(degenerate):
    weights[degenerate] = 0.0
    weights[degenerate, 0] = 1.0

  donor_y, donor_x = donors_yx[indices, 0], donors_yx[indices, 1]
  neighbor_colors = img[donor_y, donor_x]
  blended = (np.sum(neighbor_colors * weights[..., np.newaxis], axis=1)
       / np.sum(weights, axis=1, keepdims=True))

  out = img.copy()
  out[holes_yx[:, 0], holes_yx[:, 1]] = blended
  return out



def apply_exposure_gamma(
  image: Union[str, np.ndarray],
  exposure: float = 0.0,
  gamma: float = 1.0,
  shadow_bias: float = 0.0,
  output_path: Optional[str] = None
) -> np.ndarray:
  """Applies exposure and gamma, optionally weighted by the image's own luminance.

  Exposure is applied in linear light (like a camera or Photoshop's Exposure
  tool: multiplying the encoded value directly would overshoot since the
  encoding curve is itself non-linear). Gamma is a power curve on the
  encoded signal. shadow_bias blends both toward their full effect in black
  and toward no effect in white, so a shadow lift doesn't blow out
  highlights that were already fine. Both inputs are accepted either as
  arrays or as paths.

  Args:
    image (Union[str, np.ndarray]): The source image or the path to read it from.
    exposure (float): Stops (0.0 is neutral; each +/-1.0 doubles/halves the signal).
    gamma (float): Power curve on the encoded signal (1.0 is neutral; >1.0
      lifts midtones, <1.0 crushes them). Non-positive values are ignored.
    shadow_bias (float): 0-1, weights exposure and gamma by the pixel's own
      luminance -- 0.0 applies them uniformly (default); 1.0 applies their
      full effect in black, tapering to none in white.
    output_path (Optional[str]): Destination file, when the result must be saved.

  Returns:
    np.ndarray: The corrected image as a uint8 array.
  """
  # 1. Resolve the input, whichever side of the memory boundary it sits on.
  img = image if isinstance(image, np.ndarray) else load_image_as_numpy(image)

  # 2. Promote a flat buffer to the 3D layout the luminance mask expects.
  if img.ndim == 2:
    img = img[..., np.newaxis]
  if img.shape[-1] == 1:
    img = np.repeat(img, 3, axis=-1)

  # 3. Delegate to the same routine the bake pipeline uses, so this stays in
  # lockstep with it instead of drifting with a second copy of the math.
  corrected = apply_color_correction(img, {
    "exposure": exposure,
    "gamma": gamma,
    "exp_shadow_bias": shadow_bias,
  })

  # 4. Commit the result to disk when a destination is requested.
  if output_path:
    save_numpy_as_image(corrected, output_path)

  return corrected