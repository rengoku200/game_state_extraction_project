import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2

from template_match_for_model_train import detect_heroes_in_killfeed, load_templates


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_name(name: str) -> str:
    return name.replace("_normal", "").replace("_mirrored", "")


def crop_with_pad(
    img,
    x: int,
    y: int,
    w: int,
    h: int,
    pad: int,
) -> Optional["cv2.Mat"]:
    ih, iw = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def pick_best_per_side(dets: List[Dict]) -> Dict[str, Dict]:
    best: Dict[str, Dict] = {}
    for d in dets:
        side = d.get("side")
        if side not in ("killer", "victim"):
            continue
        if side not in best or d.get("score", 0) > best[side].get("score", 0):
            best[side] = d
    return best


def main() -> int:
    """Parse command line arguments and run the main logic."""

    ap = argparse.ArgumentParser(
        description="Generate hero-icon crops using template matches (auto-labeling)."
    )
    ap.add_argument("--frames-dir", required=True, help="Kill-feed crops folder.")
    ap.add_argument("--templates-dir", default="data/templates", help="Templates folder.")
    ap.add_argument(
        "--out-dir",
        default="data/icon_dataset",
        help="Output dataset root (ImageFolder style: out_dir/class/*.png).",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Only save detections with score >= this value.",
    )
    
    
    ap.add_argument(
        "--size",
        type=int,
        default=64,
        help="Final square size for saved crops (e.g. 64).",
    )
    
    args = ap.parse_args()




    templates = load_templates(args.templates_dir)
    safe_makedirs(args.out_dir)

    frame_files = [
        os.path.join(args.frames_dir, f)
        for f in sorted(os.listdir(args.frames_dir))
        if f.lower().endswith(".png")
    ]
    if not frame_files:
        print(f"No PNG frames found in: {args.frames_dir}")
        return 1

    saved_per_class: Dict[str, int] = {}

    processed = 0
    saved = 0
    for i, path in enumerate(frame_files):

        img = cv2.imread(path)
        if img is None:
            continue

        dets = detect_heroes_in_killfeed(img, templates)
        best = pick_best_per_side(dets)
        processed += 1

        for side, d in best.items():
            score = float(d.get("score", 0.0))
            if score < args.min_score:
                continue

            hero = clean_name(d["hero"])


            (t_h, t_w) = d.get("t_shape", (24, 50))
            x, y = d["location"]

            crop = crop_with_pad(img, int(x), int(y), int(t_w), int(t_h), 1)
            if crop is None:
                continue

            crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)

            class_dir = os.path.join(args.out_dir, hero)
            safe_makedirs(class_dir)
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(class_dir, f"{base}_{side}_s{score:.2f}.png")
            cv2.imwrite(out_path, crop)

            saved_per_class[hero] = saved_per_class.get(hero, 0) + 1
            saved += 1

        if processed % 200 == 0:
            print(f"Processed {processed} frames, saved {saved} crops...")

    print("\nDone.")
    print(f"  Frames processed: {processed}")
    print(f"  Crops saved     : {saved}")
    print(f"  Dataset root    : {args.out_dir}")

    top = sorted(saved_per_class.items(), key=lambda x: x[1], reverse=True)[:15]
    print("\nTop classes by saved crops:")
    for name, cnt in top:
        print(f"  {name:24s} {cnt}")

    return 0


if __name__ == "__main__":
    main()

