from __future__ import annotations
import argparse
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

INVALID_DEPTH_THRESHOLD = 65000.0


def iter_nested_items(data: Any) -> Iterable[Tuple[str, str, int, Dict[str, Any]]]:
    if isinstance(data, dict):
        for dataset, scenes in data.items():
            if isinstance(scenes, dict):
                for scene, entries in scenes.items():
                    if isinstance(entries, list):
                        for index, item in enumerate(entries):
                            if isinstance(item, dict) and item.get("id"):
                                yield dataset, scene, index, item
                    else:
                        for _, sub_scene, index, item in iter_nested_items(entries):
                            joined_scene = f"{scene}/{sub_scene}" if sub_scene else str(scene)
                            yield str(dataset), joined_scene, index, item
            elif isinstance(scenes, list):
                for index, item in enumerate(scenes):
                    if isinstance(item, dict) and item.get("id"):
                        yield str(dataset), "", index, item
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict) and item.get("id"):
                yield "", "", index, item

def count_items(data: Any) -> int:
    return sum(1 for _ in iter_nested_items(data))

def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "target"
    return text[:max_len]

def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def normalize_masks(mask_array: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    masks = np.asarray(mask_array)
    if masks.size == 0:
        return np.zeros((0, image_hw[0], image_hw[1]), dtype=bool)

    masks = np.squeeze(masks)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 4:
        masks = np.squeeze(masks, axis=1) if masks.shape[1] == 1 else masks.reshape(-1, masks.shape[-2], masks.shape[-1])

    if masks.ndim != 3:
        raise ValueError(f"Unexpected SAM mask shape after squeeze: {masks.shape}")

    if masks.shape[-2:] != image_hw:
        fixed_masks = []
        for mask in masks:
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            mask_img = mask_img.resize((image_hw[1], image_hw[0]), resample=Image.Resampling.NEAREST)
            fixed_masks.append(np.asarray(mask_img) > 0)
        return np.stack(fixed_masks, axis=0)

    return masks > 0

def box_xywh_norm_to_xyxy(box: List[float], image_width: int, image_height: int) -> List[int]:
    x, y, w, h = [float(v) for v in box]
    x1 = int(round(x * image_width))
    y1 = int(round(y * image_height))
    x2 = int(round((x + w) * image_width))
    y2 = int(round((y + h) * image_height))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1, min(image_width - 1, x2))
    y2 = max(y1, min(image_height - 1, y2))
    return [x1, y1, x2, y2]

def mask_bbox_xyxy(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def region_label(center_x: float, center_y: float) -> str:
    horizontal = "left" if center_x < 1.0 / 3.0 else "right" if center_x > 2.0 / 3.0 else "center"
    vertical = "upper" if center_y < 1.0 / 3.0 else "lower" if center_y > 2.0 / 3.0 else "middle"
    return f"{vertical}-{horizontal}"

def depth_bucket(median_depth: Optional[float], global_stats: Dict[str, Optional[float]]) -> str:
    if median_depth is None:
        return "unknown"
    p33 = global_stats.get("p33_depth")
    p66 = global_stats.get("p66_depth")
    if p33 is None or p66 is None:
        return "unknown"
    if median_depth <= p33:
        return "near"
    if median_depth >= p66:
        return "far"
    return "middle"

def visible_size_bucket(area_ratio: float) -> str:
    if area_ratio < 0.005:
        return "very small"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.08:
        return "medium"
    return "large"

def compute_mask_item(
    mask: np.ndarray,
    depth: np.ndarray,
    valid_depth_mask: np.ndarray,
    global_stats: Dict[str, Optional[float]],
    box_xywh_normalized: Optional[List[float]],
    score: Optional[float],
    idx: int,
) -> Dict[str, Any]:
    image_height, image_width = mask.shape
    mask_area = int(mask.sum())
    area_ratio = float(mask_area / max(1, image_height * image_width))
    bbox = mask_bbox_xyxy(mask)
    if bbox is None and box_xywh_normalized is not None:
        bbox = box_xywh_norm_to_xyxy(box_xywh_normalized, image_width, image_height)
    if bbox is None:
        bbox = [0, 0, 0, 0]

    ys, xs = np.where(mask)
    if xs.size > 0:
        center_x = float(xs.mean() / max(1, image_width - 1))
        center_y = float(ys.mean() / max(1, image_height - 1))
    else:
        center_x = float((bbox[0] + bbox[2]) / 2.0 / max(1, image_width - 1))
        center_y = float((bbox[1] + bbox[3]) / 2.0 / max(1, image_height - 1))

    depth_mask = mask & valid_depth_mask
    depth_values = depth[depth_mask]
    valid_depth_ratio = float(depth_mask.sum() / max(1, mask_area))
    if depth_values.size == 0:
        depth_stats = {
            "median_depth": None,
            "p25_depth": None,
            "p75_depth": None,
            "mean_depth": None,
        }
    else:
        p25, median, p75 = np.percentile(depth_values, [25, 50, 75])
        depth_stats = {
            "median_depth": float(median),
            "p25_depth": float(p25),
            "p75_depth": float(p75),
            "mean_depth": float(depth_values.mean()),
        }

    return {
        "idx": idx,
        "score": score,
        "area": mask_area,
        "area_ratio": area_ratio,
        "visible_size": visible_size_bucket(area_ratio),
        "valid_depth_ratio": valid_depth_ratio,
        "bbox_xyxy": [int(v) for v in bbox],
        "box_xywh_normalized": box_xywh_normalized,
        "center_xy_normalized": [center_x, center_y],
        "image_region": region_label(center_x, center_y),
        "relative_depth": depth_bucket(depth_stats["median_depth"], global_stats),
        **depth_stats,
    }

def choose_best_item(items: List[Dict[str, Any]], min_area_ratio: float, min_valid_depth_ratio: float) -> Optional[Dict[str, Any]]:
    if not items:
        return None

    def item_score(item: Dict[str, Any]) -> Tuple[int, float, float, float]:
        area_ok = 1 if float(item.get("area_ratio", 0.0)) >= min_area_ratio else 0
        depth_ok = 1 if float(item.get("valid_depth_ratio", 0.0)) >= min_valid_depth_ratio else 0
        prob = item.get("score")
        prob_value = float(prob) if isinstance(prob, (int, float)) else 0.0
        area_ratio = float(item.get("area_ratio", 0.0))
        return (depth_ok, area_ok, prob_value, area_ratio)

    return max(items, key=item_score)

def load_sam3_predictor(args: argparse.Namespace):
    if args.sam3_src and args.sam3_src not in sys.path:
        sys.path.insert(0, args.sam3_src)

    from sam3.model_builder import build_sam3_predictor

    return build_sam3_predictor(
        checkpoint_path=args.sam3_checkpoint,
        version=args.sam3_version,
        compile=False,
        warm_up=False,
        use_fa3=False,
        async_loading_frames=False,
        max_num_objects=args.sam3_model_max_num_objects,
        multiplex_count=args.sam3_multiplex_count,
    )

def run_sam_prompt(
    predictor: Any,
    session_id: str,
    prompt: str,
    frame_index: int,
    output_prob_thresh: float,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, List[Optional[List[float]]], List[Optional[float]]]:
    response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": prompt,
            "output_prob_thresh": output_prob_thresh,
        }
    )
    outputs = response.get("outputs", {})
    masks = normalize_masks(to_numpy(outputs.get("out_binary_masks", np.zeros((0, *image_hw), dtype=bool))), image_hw)

    boxes = outputs.get("out_boxes_xywh")
    if boxes is None:
        boxes_list = [None] * masks.shape[0]
    else:
        boxes_np = to_numpy(boxes).reshape(-1, 4)
        boxes_list = [[float(v) for v in row] for row in boxes_np[: masks.shape[0]]]
        if len(boxes_list) < masks.shape[0]:
            boxes_list.extend([None] * (masks.shape[0] - len(boxes_list)))

    probs = outputs.get("out_probs")
    if probs is None:
        probs_list = [None] * masks.shape[0]
    else:
        probs_np = to_numpy(probs).reshape(-1)
        probs_list = [float(v) for v in probs_np[: masks.shape[0]]]
        if len(probs_list) < masks.shape[0]:
            probs_list.extend([None] * (masks.shape[0] - len(probs_list)))

    return masks, boxes_list, probs_list

def save_prompt_masks(
    mask_dir: str,
    item_id: str,
    prompt: str,
    masks: np.ndarray,
) -> Optional[str]:
    if masks.shape[0] == 0:
        return None
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f"{item_id}_{sanitize_filename(prompt)}.npz")
    np.savez_compressed(mask_path, masks=masks.astype(np.uint8))
    return mask_path

def collect_prompts(extraction: Dict[str, Any]) -> Dict[str, str]:
    targets = extraction.get("targets", {}) if isinstance(extraction, dict) else {}
    subject = str(targets.get("subject", "") or "").strip()
    reference = str(targets.get("reference", "") or "").strip()
    prompts = {}
    if subject:
        prompts["subject"] = subject
    if reference and reference != subject:
        prompts["reference"] = reference
    return prompts


def add_unique_candidate(candidates: List[str], text: str) -> None:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return
    if text.lower() not in {candidate.lower() for candidate in candidates}:
        candidates.append(text)

def build_sam_query_candidates(prompt: str) -> List[str]:
    candidates: List[str] = []
    normalized = re.sub(r"\s+", " ", prompt.strip())
    add_unique_candidate(candidates, normalized)

    simplified = re.sub(r"^(the|a|an)\s+", "", normalized, flags=re.IGNORECASE)
    add_unique_candidate(candidates, simplified)

    modifier_splits = [
        " with ",
        " featuring ",
        " including ",
        " containing ",
        " covered with ",
        " located ",
        " on the ",
        " in the ",
        " near the ",
    ]
    for split_token in modifier_splits:
        for source in [normalized, simplified]:
            lowered = source.lower()
            index = lowered.find(split_token)
            if index > 0:
                add_unique_candidate(candidates, source[:index])

    words = simplified.split()
    if len(words) > 4:
        add_unique_candidate(candidates, " ".join(words[-4:]))
        add_unique_candidate(candidates, " ".join(words[-3:]))
    return candidates
