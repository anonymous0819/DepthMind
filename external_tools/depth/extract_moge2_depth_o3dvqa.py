"""Batch MoGe-V2 depth + estimated intrinsics extraction for O3DVQA MCQ test.

Reads unique image paths from `test_qa_mcq.json`, runs MoGe-V2 with NO
GT intrinsics (K is estimated internally), and saves per-image .npz files
containing:
    depth       : (H, W) float32, metric meters, invalid pixels -> 0
    intrinsics  : (3, 3) float32, PIXEL-space K (converted from MoGe's
                  normalized K by multiplying with W / H)
    valid_mask  : (H, W) uint8, 1 where depth is finite & positive
    K_norm      : (3, 3) float32, the raw normalized K from MoGe (kept
                  for debugging)

Output layout (mirrors estimated_depth_da3nested/):
    <out_root>/<dataset>/<scene>/rgb/<stem>.npz

The test_qa_mcq.json uses Windows-style paths (e.g. "rgb\\006_RGBVis_0.png");
resolution is handled by the local o3dvqa_extraction_common helpers.

Usage:
    USE_TF=0 CUDA_VISIBLE_DEVICES=0 python \
        /root/src/UAV_reasoning/DepthMind/external_tools/depth/extract_moge2_depth_o3dvqa.py \
        --qa_file /root/src/UAV_reasoning/O3DVQA_v2/test_qa_mcq.json \
        --out_root /root/src/UAV_reasoning/DepthMind/estimated_depth_moge2nested
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

DEFAULT_MOGE_SRC = "/root/src/UAV_reasoning/MoGe-main"
DEFAULT_MOGE_CHECKPOINT = "/root/src/UAV_reasoning/MoGE2/model.pt"
DEFAULT_DATA_DIR = "/root/src/UAV_reasoning/O3DVQA_v2"


def collect_unique_images(qa_file: str, data_dir: str) -> List[Tuple[str, str, str, str]]:
    """Return list of (dataset, scene, image_path_relative, image_path_abs)."""
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


def build_moge2(device_str: str, moge_src: str, checkpoint: str, use_fp16: bool):
    if moge_src not in sys.path:
        sys.path.insert(0, moge_src)
    from moge.model.v2 import MoGeModel

    device = torch.device(device_str)
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()
    if use_fp16 and device.type == "cuda":
        model = model.half()

    # Monkey-patch forward so we can capture the RAW sigmoid mask probability
    # (MoGeModel.infer() only exposes the binarised mask; the continuous
    # sigmoid mask is what we want for depth-confidence gating).
    _orig_forward = MoGeModel.forward

    def _forward_capture(self, image, num_tokens):
        out = _orig_forward(self, image, num_tokens)
        raw_mask = out.get("mask", None)  # (B, H, W) sigmoid prob in [0, 1]
        self._last_raw_mask = raw_mask.detach().float() if raw_mask is not None else None
        return out

    if not getattr(MoGeModel, "_forward_patched_for_conf", False):
        MoGeModel.forward = _forward_capture
        MoGeModel._forward_patched_for_conf = True
    return model, device


def run_moge2(model: Any, device: torch.device, image_path: str,
              resolution_level: int, num_tokens: int | None,
              use_fp16: bool) -> Dict[str, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_np, dtype=torch.float32,
                                device=device).permute(2, 0, 1)
    # reset raw mask cache so we can tell whether the forward really wrote
    model._last_raw_mask = None
    with torch.inference_mode():
        out = model.infer(
            image_tensor,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16 and device.type == "cuda",
        )
    # depth: (H, W) or (H, W) with inf at masked pixels
    depth = out["depth"].detach().float().cpu().numpy().astype(np.float32)
    if depth.shape != (H, W):
        # In case MoGe returns model-resolution size (unlikely, but be safe)
        from PIL import Image as _PI
        # Replace inf with 0 first
        depth_safe = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        depth_safe = np.asarray(
            _PI.fromarray(depth_safe, mode="F").resize((W, H), _PI.Resampling.BILINEAR),
            dtype=np.float32,
        )
        depth = depth_safe
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)

    # intrinsics: MoGe returns NORMALIZED K: cx=0.5, cy=0.5, fx/fy in [0, 1]-ish
    K_norm = out["intrinsics"].detach().float().cpu().numpy().astype(np.float32)
    if K_norm.shape != (3, 3):
        K_norm = K_norm.reshape(3, 3)

    # Convert normalized K to PIXEL K.
    # MoGe convention: image is normalized by max(W, H)? Actually looking at its
    # infer() code the intrinsics are built with center (0.5, 0.5) and focal
    # in NORMALIZED units where the image size is (1, 1) after scaling by
    # image dimensions. The standard convention used throughout MoGe utils3d
    # is: pixel_x = fx * W * X/Z + cx * W. So we multiply row0 by W and row1 by H.
    K_pix = K_norm.copy().astype(np.float32)
    K_pix[0, :] *= float(W)
    K_pix[1, :] *= float(H)

    # Grab the raw sigmoid mask captured by the monkey-patched forward.
    # MoGe2 batchifies internally, so we get shape (1, H_in, W_in) or
    # (H_in, W_in). infer() feeds the image AS-IS so H_in==H, W_in==W.
    raw_mask = getattr(model, "_last_raw_mask", None)
    if raw_mask is not None:
        rm = raw_mask.cpu().numpy()
        if rm.ndim == 3:
            rm = rm[0]
        if rm.shape != (H, W):
            from PIL import Image as _PI
            rm = np.asarray(
                _PI.fromarray(rm.astype(np.float32), mode="F")
                .resize((W, H), _PI.Resampling.BILINEAR),
                dtype=np.float32,
            )
        mask_prob = rm.astype(np.float16)  # (H, W), 0-1 sigmoid prob
    else:
        mask_prob = None

    return {
        "depth": depth,
        "intrinsics": K_pix,
        "K_norm": K_norm,
        "valid_mask": valid.astype(np.uint8),
        "mask_prob": mask_prob,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file",
                        default="/root/src/UAV_reasoning/O3DVQA_v2/test_qa_mcq.json")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_root",
                        default=str(ROOT / "estimated_depth_moge2nested"))
    parser.add_argument("--moge_src", default=DEFAULT_MOGE_SRC)
    parser.add_argument("--pretrained", default=DEFAULT_MOGE_CHECKPOINT)
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--num_tokens", type=int, default=0,
                        help="0 -> None (auto)")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh_conf", action="store_true",
                        help="Re-run inference to add mask_prob field to "
                             "existing npz files that lack it (keeps depth/K "
                             "identical, only appends mask_prob).")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    images = collect_unique_images(args.qa_file, args.data_dir)
    print(f"[moge2] {len(images)} unique images in {args.qa_file}")
    if args.max_samples > 0:
        images = images[: args.max_samples]
        print(f"[moge2] limited to first {len(images)}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    todo = []
    for ds, sub, rel, abs_p in images:
        stem = Path(rel.replace("\\", "/")).stem
        target = out_root / ds / sub / "rgb" / f"{stem}.npz"
        if target.exists() and not args.overwrite:
            if args.refresh_conf:
                # Check whether the existing npz already has mask_prob
                try:
                    with np.load(target) as npz:
                        if "mask_prob" in npz.files:
                            continue
                except Exception:
                    pass
                todo.append((ds, sub, rel, abs_p, target))
            continue
        todo.append((ds, sub, rel, abs_p, target))
    print(f"[moge2] {len(images) - len(todo)} already done, {len(todo)} to process")
    if not todo:
        print("[moge2] nothing to do")
        return

    num_tokens = None if args.num_tokens == 0 else int(args.num_tokens)
    model, device = build_moge2(args.device, args.moge_src, args.pretrained, args.fp16)
    print(f"[moge2] MoGe-V2 loaded on {device}, fp16={args.fp16}, "
          f"resolution_level={args.resolution_level}, num_tokens={num_tokens}")

    t0 = time.time()
    for i, (ds, sub, rel, abs_p, target) in enumerate(tqdm(todo, desc="MoGe2")):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp.npz")
        try:
            payload = run_moge2(model, device, abs_p,
                                resolution_level=args.resolution_level,
                                num_tokens=num_tokens,
                                use_fp16=args.fp16)
            save_kwargs = dict(
                depth=payload["depth"],
                intrinsics=payload["intrinsics"],
                K_norm=payload["K_norm"],
                valid_mask=payload["valid_mask"],
                source_image=abs_p,
            )
            if payload.get("mask_prob") is not None:
                save_kwargs["mask_prob"] = payload["mask_prob"]
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
            print(f"[moge2] {i+1}/{len(todo)} avg={avg:.2f}s/frame eta={eta/60:.1f}min")

    print(f"[moge2] done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
