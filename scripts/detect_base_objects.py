"""Run keremberke/yolov5s-clash-of-clans on downloaded base images.

For each image:
- Save an annotated JPG to {out_dir}/annotated/{relative_path}
- Append a structured record to {out_dir}/detections.jsonl with bbox/score/class.

Also writes {out_dir}/summary.json with overall class counts so we can quickly
see what the model is and isn't picking up.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import colorsys

import numpy as np
import torch  # noqa: F401  (imported for the side-effect of patching torch.load below)
import torch.serialization as _ts  # noqa: F401
from PIL import Image, ImageDraw, ImageFont

# PyTorch 2.6 flipped torch.load default to weights_only=True, which rejects
# legacy YOLOv5 checkpoints. Force weights_only=False so we can load the model.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat  # type: ignore[assignment]

import yolov5  # type: ignore  # noqa: E402


def collect_images(root: Path, limit: int | None) -> list[Path]:
    paths = sorted(p for p in root.rglob("*.jpg"))
    if limit:
        paths = paths[:limit]
    return paths


def _class_colors(names: dict) -> dict[int, tuple[int, int, int]]:
    n = max(1, len(names))
    out: dict[int, tuple[int, int, int]] = {}
    for i in names:
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 0.95)
        out[i] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def annotate(
    src: Path, dst: Path, detections: list[dict], colors: dict[int, tuple[int, int, int]]
) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = colors.get(d["class_id"], (255, 0, 0))
            for w in range(3):
                draw.rectangle((x1 - w, y1 - w, x2 + w, y2 + w), outline=color)
            label = f"{d['class_name']} {d['score']:.2f}"
            tb = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=color)
            draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("runs/base_layouts/th4_images"))
    ap.add_argument("--out", type=Path, default=Path("runs/base_layouts/th4_detections"))
    ap.add_argument("--model", default="keremberke/yolov5s-clash-of-clans")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.out / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.src, args.limit)
    print(f"Found {len(images)} images under {args.src}", file=sys.stderr)
    if not images:
        sys.exit("no images")

    print(f"Loading {args.model} ...", file=sys.stderr)
    model = yolov5.load(args.model)
    model.conf = args.conf
    model.iou = args.iou
    model.max_det = 200
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"Model classes ({len(names)}): {list(names.values())}", file=sys.stderr)
    colors = _class_colors(names)

    cls_counter: Counter = Counter()
    per_image_counts: list[int] = []
    jsonl_path = args.out / "detections.jsonl"
    with jsonl_path.open("w") as jl:
        for i, img_path in enumerate(images, 1):
            results = model(str(img_path), size=args.imgsz)
            pred = results.pred[0]  # tensor [N, 6]: x1,y1,x2,y2,score,cls
            dets = []
            for *xyxy, score, cls in pred.tolist():
                cls_id = int(cls)
                dets.append(
                    {
                        "bbox": [float(v) for v in xyxy],
                        "score": float(score),
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                    }
                )
                cls_counter[names.get(cls_id, str(cls_id))] += 1
            per_image_counts.append(len(dets))

            rel = img_path.relative_to(args.src)
            try:
                h, w = results.ims[0].shape[:2]  # type: ignore[attr-defined]
            except Exception:
                h = w = None
            jl.write(
                json.dumps(
                    {
                        "image": str(rel),
                        "width": int(w) if w else None,
                        "height": int(h) if h else None,
                        "detections": dets,
                    }
                )
                + "\n"
            )

            ann_dest = annotated_dir / rel
            annotate(img_path, ann_dest, dets, colors)

            if i % 10 == 0 or i == len(images):
                avg = sum(per_image_counts) / max(1, len(per_image_counts))
                print(
                    f"  [{i}/{len(images)}] {rel}  dets={len(dets)}  avg/img={avg:.2f}",
                    file=sys.stderr,
                )

    summary = {
        "model": args.model,
        "image_count": len(images),
        "total_detections": sum(per_image_counts),
        "avg_detections_per_image": sum(per_image_counts) / max(1, len(per_image_counts)),
        "images_with_zero_detections": sum(1 for c in per_image_counts if c == 0),
        "class_counts": dict(cls_counter.most_common()),
        "params": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
