"""Runs GenieSAM's SAM3 text-prompt facial segmentation and saves one mask PNG per category.

Meant to be invoked as a subprocess using the separate `geniesam` conda env's own python.exe
(see texture_eyebrow_sam3.py) — GenieSAM's dependencies (torch, a git-installed `sam3`
package, opencv<4.10) conflict with this project's own pins, so this script never runs
inside the mask_luminance_tool env; it only ever runs standalone, in-process there.

Two modes, sharing one model load:
  --image IN --output-dir OUT           single image -> OUT/<category>.png
  --image-dir IN_DIR --output-root ROOT  every image in IN_DIR -> ROOT/<stem>/<category>.png
Batch mode exists because loading the ~3GB SAM3 checkpoint dominates runtime — looping single
mode over many images would reload it every time, whereas batch mode loads it once and reuses
it for every image in IN_DIR.

Mirrors the exact call sequence GenieSAM's own inference.py::run_job uses (load_sam3_model ->
load_image -> segment_image -> resize_masks_to_image), including the cuda SDP-backend
workaround and bfloat16 autocast, so results match GenieSAM's own service.
"""
import argparse
import os
import sys

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Single input image path.")
    mode.add_argument("--image-dir", help="Directory of input images (batch mode: one model load for all of them).")
    p.add_argument("--output-dir", help="Single mode: directory to write one PNG per found category.")
    p.add_argument("--output-root", help="Batch mode: masks written to <output-root>/<image-stem>/<category>.png.")
    p.add_argument("--checkpoint", required=True, help="Path to the SAM3 .pth checkpoint.")
    p.add_argument("--geniesam-repo", required=True, help="Path to the GenieSAM repo (added to sys.path).")
    p.add_argument("--categories", nargs="+", required=True, help="GenieSAM text-prompt category names.")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument(
        "--beard-score-threshold",
        type=float,
        default=None,
        help="Override GenieSAM config.yaml's beard_score threshold (0-1). Lower catches more "
             "beard on textures where hair/skin color contrast is low (e.g. darker skin tones), "
             "at the cost of more false positives. Defaults to the GenieSAM config's own value.",
    )
    p.add_argument(
        "--eyebrow-score-threshold",
        type=float,
        default=None,
        help="Score threshold (0-1) for Left/Right Eyebrow only, replacing config.yaml's global "
             "score threshold for those two categories. Lower catches faint/low-contrast painted "
             "brows (e.g. lighter textures), at the cost of more false positives.",
    )
    args = p.parse_args()
    if args.image and not args.output_dir:
        p.error("--image requires --output-dir")
    if args.image_dir and not args.output_root:
        p.error("--image-dir requires --output-root")
    return args


def _safe_filename(category: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in category) + ".png"


def _install_category_thresholds(seg_module, overrides: dict) -> None:
    """Gives each category in ``overrides`` its own score threshold, in place of the global one.

    GenieSAM's segment_image drops any category scoring below the global score_threshold
    BEFORE its beard-specific check, so a per-category threshold below the global one never
    takes effect through its own arguments (e.g. beard_score_threshold=0.6 with score=0.8
    still drops a 0.7 beard). Wrapping segment_with_text — which segment_image looks up as a
    module global — to report pass/fail scores for the overridden categories keeps everything
    in one segment_image call, so GenieSAM's cross-category post-processing is unchanged.
    """
    if not overrides:
        return
    base = seg_module.segment_with_text

    def segment_with_text(model, backbone_out, text_prompt, device, image_size):
        mask, score = base(model, backbone_out, text_prompt, device, image_size)
        threshold = overrides.get(text_prompt)
        if threshold is not None:
            score = 1.0 if score >= threshold else 0.0
        return mask, score

    seg_module.segment_with_text = segment_with_text


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, args.geniesam_repo)

    import cv2
    import torch
    import segmentation.segmentation as seg_module
    from config_loader import get_config
    from segmentation.segmentation import load_image, load_sam3_model, resize_masks_to_image, segment_image

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    if args.device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    model, image_transform = load_sam3_model(args.device, args.image_size, sam3_ckpt=args.checkpoint)
    cfg = get_config()
    beard_score_threshold = cfg.beard_score_threshold
    overrides = {}
    if args.beard_score_threshold is not None:
        overrides["Beard"] = args.beard_score_threshold
    if args.eyebrow_score_threshold is not None:
        overrides["Left Eyebrow"] = overrides["Right Eyebrow"] = args.eyebrow_score_threshold
    _install_category_thresholds(seg_module, overrides)

    def segment_one(image_path: str, out_dir: str) -> list:
        image_np = load_image(image_path)
        if image_np is None:
            print(f"Failed to load image: {image_path}", file=sys.stderr)
            return []
        segment_kwargs = dict(
            model=model,
            image_transform=image_transform,
            image_np=image_np,
            categories=args.categories,
            device=args.device,
            image_size=args.image_size,
            min_area=cfg.min_area,
            score_threshold=cfg.score_threshold,
            area_threshold=cfg.area_threshold,
            beard_score_threshold=beard_score_threshold,
            post_process=cfg.post_process_enabled,
        )
        if args.device == "cuda":
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks = segment_image(**segment_kwargs)
        else:
            with torch.no_grad():
                masks = segment_image(**segment_kwargs)
        masks = resize_masks_to_image(masks, image_np.shape[:2])

        os.makedirs(out_dir, exist_ok=True)
        found = []
        for category in args.categories:
            mask = masks.get(category)
            if mask is None:
                continue
            cv2.imwrite(os.path.join(out_dir, _safe_filename(category)), mask.astype("uint8") * 255)
            found.append(category)
        return found

    if args.image:
        found = segment_one(args.image, args.output_dir)
        if not found:
            print(f"No categories found above threshold: {args.categories}", file=sys.stderr)
            return 1
        print(f"Saved: {found}")
        return 0

    image_files = sorted(f for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in _IMAGE_EXTS)
    if not image_files:
        print(f"No images found in: {args.image_dir}", file=sys.stderr)
        return 1

    any_found = False
    for filename in image_files:
        stem = os.path.splitext(filename)[0]
        found = segment_one(os.path.join(args.image_dir, filename), os.path.join(args.output_root, stem))
        print(f"{filename}: {'ok ' + str(found) if found else 'no categories found'}")
        any_found = any_found or bool(found)

    return 0 if any_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
