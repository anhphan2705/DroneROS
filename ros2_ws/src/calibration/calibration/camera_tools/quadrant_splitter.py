#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import cv2


def split_quadrants(img):
    """
    Split into 4 even-sized quadrants exactly like:
      half_w = (w//2) & ~1
      half_h = (h//2) & ~1

    Returns: (cam0, cam1, cam2, cam3) as BGR uint8 images
      cam0 = TL, cam1 = TR, cam2 = BL, cam3 = BR
    """
    h, w = img.shape[:2]
    half_w = (w // 2) & ~1
    half_h = (h // 2) & ~1

    # If input is odd-sized, we simply ignore the leftover row/col.
    w_used = half_w * 2
    h_used = half_h * 2
    img = img[:h_used, :w_used]

    cam0 = img[0:half_h,        0:half_w       ]  # TL
    cam1 = img[0:half_h,        half_w:w_used  ]  # TR
    cam2 = img[half_h:h_used,   0:half_w       ]  # BL
    cam3 = img[half_h:h_used,   half_w:w_used  ]  # BR
    return cam0, cam1, cam2, cam3


def main():
    ap = argparse.ArgumentParser(
        description="Split mosaic images into cam0..cam3 quadrants (TL,TR,BL,BR) matching VPI ROI logic."
    )
    ap.add_argument("--input_dir", required=True, help="Folder containing .png/.jpg mosaics")
    ap.add_argument("--output_dir", default=".", help="Where to create camera0..camera3 folders")
    ap.add_argument("--pattern", default="*.png,*.jpg,*.jpeg,*.bmp", help="Comma-separated glob patterns")
    ap.add_argument("--recursive", action="store_true", help="Recursive glob")
    ap.add_argument("--png", action="store_true", help="Force output to .png (recommended for your calibrator)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    # Create output folders named camera0..camera3 (matches your calibrator example usage)
    cam_dirs = []
    for i in range(4):
        d = out_dir / f"camera{i}"
        d.mkdir(parents=True, exist_ok=True)
        cam_dirs.append(d)

    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]
    paths = []
    for pat in patterns:
        if args.recursive:
            paths.extend(glob.glob(str(in_dir / "**" / pat), recursive=True))
        else:
            paths.extend(glob.glob(str(in_dir / pat)))

    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"No images found in {in_dir} with pattern(s): {args.pattern}")

    print(f"Found {len(paths)} images")

    for idx, p in enumerate(paths, 1):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read: {p}")
            continue

        cam0, cam1, cam2, cam3 = split_quadrants(img)

        src = Path(p)
        stem = src.stem
        ext = ".png" if args.png else src.suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            ext = ".png"  # safe default

        outs = [
            cam_dirs[0] / f"{stem}{ext}",
            cam_dirs[1] / f"{stem}{ext}",
            cam_dirs[2] / f"{stem}{ext}",
            cam_dirs[3] / f"{stem}{ext}",
        ]

        if (not args.overwrite) and any(o.exists() for o in outs):
            print(f"[SKIP] Exists (use --overwrite): {src.name}")
            continue

        cv2.imwrite(str(outs[0]), cam0)
        cv2.imwrite(str(outs[1]), cam1)
        cv2.imwrite(str(outs[2]), cam2)
        cv2.imwrite(str(outs[3]), cam3)

        if idx % 50 == 0 or idx == len(paths):
            print(f"Processed {idx}/{len(paths)}")

    print("Done.")
    print(f"Output folders: {out_dir/'camera0'}, {out_dir/'camera1'}, {out_dir/'camera2'}, {out_dir/'camera3'}")
    print("Your calibrator expects *.png, so run with --png if needed.")


if __name__ == "__main__":
    main()


# Example: python3 split_quadrants.py --input_dir /path/to/mosaics --output_dir /path/to/out --png
