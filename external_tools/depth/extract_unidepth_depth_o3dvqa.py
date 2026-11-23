"""Batch UniDepth-V2 depth + estimated intrinsics extraction for O3DVQA MCQ test.

UniDepth-V2 (vitl14 default) will predict both metric depth AND camera K when
called with `camera=None`, giving us a fully GT-free depth+K pair.

Output layout (mirrors estimated_depth_da3nested/):
    <out_root>/<dataset>/<scene>/rgb/<stem>.npz
each .npz contains:
    depth       : (H, W) float32, metric meters
    intrinsics  : (3, 3) float32, PIXEL-space K estimated by UniDepth
    valid_mask  : (H, W) uint8

Usage:
    USE_TF=0 CUDA_VISIBLE_DEVICES=0 python \
        /root/src/UAV_reasoning/DepthMind/external_tools/depth/extract_unidepth_depth_o3dvqa.py \
        --qa_file /root/src/UAV_reasoning/O3DVQA_v2/test_qa_mcq.json \
        --out_root /root/src/UAV_reasoning/DepthMind/estimated_depth_unidepthnested
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from external_tools.o3dvqa_extraction_common import build_image_path
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_UNIDEPTH_SRC = "/root/src/UAV_reasoning/UniDepth-main"
DEFAULT_UNIDEPTH_WEIGHT = "/root/src/UAV_reasoning/UniDepth-main/UniDepthv2"
DEFAULT_DATA_DIR = "/root/src/UAV_reasoning/O3DVQA_v2"


def collect_unique_images(qa_file: str, data_dir: str) -> List[Tuple[str, str, str, str]]:
    with open(qa_file, "r", encoding="utf-8") as f:
        qa = json.load(f)
    seen: Dict[Tuple[str, str, str], str] = {}
    for ds, subs in qa.items():
        if not isinstance(subs, dict):
            continue
        for sub, items in subs.items():
            if not isinstance(items, list):
                continue
            for it in items:
                info = it.get("image_info") or {}
                rel = info.get("image_path")
                if not rel:
                    continue
                key = (ds, sub, rel)
                if key in seen:
                    continue
                try:
                    abs_p = build_image_path(data_dir, ds, sub, info)
                except Exception:
                    continue
                if not os.path.exists(abs_p):
                    continue
                seen[key] = abs_p
    out = []
    for (ds, sub, rel), abs_p in seen.items():
        out.append((ds, sub, rel, abs_p))
    out.sort()
    return out


def build_unidepth(device_str: str, src_dir: str, weight_dir: str):
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from unidepth.models import UniDepthV2

    device = torch.device(device_str)
    model = UniDepthV2.from_pretrained(weight_dir)
    model.interpolation_mode = "bilinear"
    model.resolution_level = 9
    model = model.to(device).eval()
    return model, device


def run_unidepth(model: Any, device: torch.device,
                 image_path: str) -> Dict[str, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    rgb = np.asarray(image).copy()
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).to(device)

    with torch.inference_mode():
        pred = model.infer(rgb_torch, camera=None)  # camera=None -> self-estimate K

    depth = pred["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
    if depth.shape != (H, W):
        from PIL import Image as _PI
        depth_safe = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        depth_safe = np.asarray(
            _PI.fromarray(depth_safe, mode="F").resize((W, H), _PI.Resampling.BILINEAR),
            dtype=np.float32,
        )
        depth = depth_safe
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)

    K = pred["intrinsics"].squeeze().detach().cpu().numpy().astype(np.float32)
    if K.shape != (3, 3):
        K = K.reshape(3, 3)

    # UniDepthV2 natively exposes a per-pixel confidence map trained with
    # NLL-style loss (higher -> depth prediction is more reliable).
    conf_t = pred.get("confidence", None)
    if conf_t is not None:
        conf = conf_t.squeeze().detach().float().cpu().numpy()
        if conf.shape != (H, W):
            from PIL import Image as _PI
            conf = np.asarray(
                _PI.fromarray(conf.astype(np.float32), mode="F")
                .resize((W, H), _PI.Resampling.BILINEAR),
                dtype=np.float32,
            )
        confidence = conf.astype(np.float16)  # (H, W)
    else:
        confidence = None

    return {
        "depth": depth,
        "intrinsics": K,
        "valid_mask": valid.astype(np.uint8),
        "confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file",
                        default="/root/src/UAV_reasoning/O3DVQA_v2/test_qa_mcq.json")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_root",
                        default=str(ROOT / "estimated_depth_unidepthnested"))
    parser.add_argument("--unidepth_src", default=DEFAULT_UNIDEPTH_SRC)
    parser.add_argument("--weight_dir", default=DEFAULT_UNIDEPTH_WEIGHT)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh_conf", action="store_true",
                        help="Re-run inference on frames whose npz lacks the "
                             "'confidence' field, keeping depth/K/valid_mask "
                             "of the fresh run.")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    images = collect_unique_images(args.qa_file, args.data_dir)
    print(f"[unidepth] {len(images)} unique images in {args.qa_file}")
    if args.max_samples > 0:
        images = images[: args.max_samples]
        print(f"[unidepth] limited to first {len(images)}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    todo = []
    for ds, sub, rel, abs_p in images:
        stem = Path(rel.replace("\\", "/")).stem
        target = out_root / ds / sub / "rgb" / f"{stem}.npz"
        if target.exists() and not args.overwrite:
            if args.refresh_conf:
                try:
                    with np.load(target) as npz:
                        if "confidence" in npz.files:
                            continue
                except Exception:
                    pass
                todo.append((ds, sub, rel, abs_p, target))
            continue
        todo.append((ds, sub, rel, abs_p, target))
    print(f"[unidepth] {len(images) - len(todo)} already done, {len(todo)} to process")
    if not todo:
        print("[unidepth] nothing to do")
        return

    model, device = build_unidepth(args.device, args.unidepth_src, args.weight_dir)
    print(f"[unidepth] UniDepth-V2 loaded on {device}, K estimation ON")

    t0 = time.time()
    for i, (ds, sub, rel, abs_p, target) in enumerate(tqdm(todo, desc="UniDepth")):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp.npz")
        try:
            payload = run_unidepth(model, device, abs_p)
            save_kwargs = dict(
                depth=payload["depth"],
                intrinsics=payload["intrinsics"],
                valid_mask=payload["valid_mask"],
                source_image=abs_p,
            )
            if payload.get("confidence") is not None:
                save_kwargs["confidence"] = payload["confidence"]
            np.savez_compressed(tmp, **save_kwargs)
            os.replace(tmp, target)
        except Exception as exc:
            import traceback
            print(f"[WARN] {abs_p}: {exc!r}")
            traceback.print_exc()
            if tmp.exists():
                tmp.unlink()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            eta = avg * (len(todo) - i - 1)
            print(f"[unidepth] {i+1}/{len(todo)} avg={avg:.2f}s/frame eta={eta/60:.1f}min")

    print(f"[unidepth] done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
