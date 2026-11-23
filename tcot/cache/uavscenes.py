"""UAVScenes-Metric v2 tool-first prompts and predicted-bbox geometry.

Objects are pre-bound by ID in the external predicted bbox cache, as in the
historical UAVScenes backend. Generated target strings are not re-grounded.
No QA metadata, GT boxes, GT intrinsics or GT depth are used by geometry.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..answer_prompts import _strip_letter_only_tail
from drope_common import load_depth
from uavscenes_common import read_predictions
from ..tool_schema import ToolCall, ToolResult

TOOL_EGO_DIST = "ego_distance_computation"
TOOL_OBJ_SIZE = "object_size_computation"
TOOL_OBJ_OBJ_DIST = "object_to_object_distance_computation"
TOOL_NAMES = {TOOL_EGO_DIST, TOOL_OBJ_SIZE, TOOL_OBJ_OBJ_DIST}
TASK_TO_TOOL = {"abs_distance_ego": TOOL_EGO_DIST, "abs_size_horiz": TOOL_OBJ_SIZE, "abs_distance_obj": TOOL_OBJ_OBJ_DIST}
TASK_TO_QNAME = {"abs_distance_ego": "distance2agent", "abs_size_horiz": "width_data", "abs_distance_obj": "distance_data"}
TOOL_FIRST_SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant answering multiple-choice "
    "questions about outdoor imagery. Each question has options labelled "
    "A, B, C or D. This conversation has two turns: (1) request exactly "
    "one external geometry measurement via a <tool_call> block (do NOT "
    "answer yet), (2) reason step by step using canonical real-world "
    "object sizes as anchors, and combine your estimate with the tool "
    "measurement to pick the final letter."
)
TOOL_MENU = (
    "Available geometry functions (choose exactly one):\n"
    "  1. ego_distance_computation(target)\n"
    "     -> Estimates the metric distance from the camera to `target`.\n"
    "  2. object_size_computation(target)\n"
    "     -> Estimates the metric physical size of `target` along the axis "
    "asked by the question.\n"
    "  3. object_to_object_distance_computation(target_a, target_b)\n"
    "     -> Estimates the metric 3D distance between two objects.\n"
    "\n"
    "Emit exactly one function call inside a <tool_call> ... </tool_call> "
    "block. The payload must be a single JSON object. Examples:\n"
    "  <tool_call>{\"name\": \"ego_distance_computation\", "
    "\"target\": \"the red car in the middle\"}</tool_call>\n"
    "  \\<tool_call\\>{\"name\": \"object_size_computation\", "
    "\"target\": \"the tall building on the left\"}</tool_call>\n"
    "  <tool_call>{\"name\": \"object_to_object_distance_computation\", "
    "\"target_a\": \"...\", \"target_b\": \"...\"}</tool_call>\n"
    "Do not output anything else in this turn."
)


def build_tool_selection_prompt(question):
    question = _strip_letter_only_tail(question or "")
    return (
        f"Question:\n{question}\n\n"
        "First, request one external geometry measurement to help you "
        "answer this multiple-choice question. Pick the function that best "
        "matches what the question asks -- keywords such as 'distance to "
        "the camera', 'size / width / height of', 'distance between' should "
        "guide your choice.\n\n"
        + TOOL_MENU
        + "\n\n### RESPONSE FORMAT (MANDATORY):\n"
        "Emit exactly one <tool_call>...</tool_call> block. Do NOT write any "
        "reasoning, do NOT write A/B/C/D, do NOT answer the question in this "
        "turn. A letter or free-form answer will be rejected."
    )


def parse_tool_call(text):
    if not text:
        return None, "no_call"
    candidates = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL | re.IGNORECASE)
    if not candidates:
        candidates = next(([payload] for payload in re.findall(r"\{[^{}]*\}", text, re.DOTALL)
                           if any(name in payload for name in TOOL_NAMES)), [])
    if not candidates:
        return None, "no_call"
    unknown = False
    for payload in candidates:
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) or name not in TOOL_NAMES:
            unknown = True
            continue
        if name == TOOL_OBJ_OBJ_DIST:
            a, b = obj.get("target_a") or obj.get("targetA"), obj.get("target_b") or obj.get("targetB")
            if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip():
                return None, "missing_target"
            return ToolCall(name, target_a=a.strip(), target_b=b.strip()), "ok"
        target = obj.get("target")
        if not isinstance(target, str) or not target.strip():
            return None, "missing_target"
        return ToolCall(name, target=target.strip()), "ok"
    return None, "unknown_name" if unknown else "bad_json"


def bbox_depth_median(depth, bbox):
    height, width = depth.shape
    x0, y0, x1, y1 = bbox
    x0, x1 = max(0, int(round(x0))), min(width, int(round(x1)))
    y0, y1 = max(0, int(round(y0))), min(height, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    for half in (2, 5, 10, 20):
        patch = depth[max(y0, cy - half):min(y1, cy + half + 1), max(x0, cx - half):min(x1, cx + half + 1)]
        values = patch[(patch > .05) & np.isfinite(patch)]
        if values.size >= 4:
            return float(np.median(values))
    patch = depth[y0:y1, x0:x1]
    values = patch[(patch > .05) & np.isfinite(patch)]
    return float(np.median(values)) if values.size else 0.


def back_project(x, y, z, k):
    return np.array([(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z], dtype=float)


class UavscenesToolBackend:
    def __init__(self, bbox_file, depth_root, intrinsics_root, data_dir=None):
        self.depth_root = Path(depth_root).resolve()
        self.intrinsics_root = Path(intrinsics_root).resolve()
        self.data_root = Path(data_dir).resolve() if data_dir is not None else None
        self.bboxes = {row["id"]: row for row in read_predictions([bbox_file])}
        if any(row.get("uses_gt") is True for row in self.bboxes.values()):
            raise ValueError("GT perception assets cannot be used as geometric tool inputs")

    def _load(self, sample):
        scene, stem = sample["scene"], Path(sample["image_path"].replace("\\", "/")).stem
        candidates = [self.depth_root / scene / (stem + suffix) for suffix in (".npy", ".npz")]
        path = next((path for path in candidates if path.is_file()), candidates[0])
        depth = load_depth(path)
        entry = self.bboxes.get(sample["id"], {})
        if entry.get("depth_unit", "m") != "m":
            raise ValueError("Tool depth must be metric meters")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "cache_config" in data:
                    config = json.loads(str(data["cache_config"].item()))
                    if not isinstance(config, dict) or config.get("uses_gt") is not False or config.get("depth_unit") != "m":
                        raise ValueError("Tool cache must declare predicted metric depth")
                if "depth_unit" in data and str(data["depth_unit"].item()) != "m":
                    raise ValueError("Tool depth must be metric meters")
        if self.data_root is not None:
            from PIL import Image
            rgb = self.data_root / f"interval5_{scene}" / "interval5_CAM" / Path(sample["image_path"].replace("\\", "/")).name
            with Image.open(rgb) as image:
                if depth.shape != (image.height, image.width):
                    raise ValueError("Tool depth must match the original pixel bbox canvas")
        k = np.asarray(np.load(self.intrinsics_root / scene / (stem + ".K.npy"), allow_pickle=False), dtype=float)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] < 2 or k[1, 1] < 2
                or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-6 or abs(k[1, 0]) > 1e-6):
            raise ValueError("Invalid predicted pixel intrinsics")
        return depth, k

    @staticmethod
    def _bbox(value, shape=None):
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.shape != (4,) or not np.isfinite(array).all() or array[2] <= array[0] or array[3] <= array[1]:
            return None
        bbox = tuple(int(round(x)) for x in array)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        if shape is not None and not (0 <= bbox[0] < bbox[2] <= shape[1] and 0 <= bbox[1] < bbox[3] <= shape[0]):
            raise ValueError("Predicted bbox is outside the tool depth canvas")
        return bbox

    def call(self, sample, call):
        name = call.name
        if name not in TOOL_NAMES:
            return ToolResult(name, None, status="invalid_call")
        entry = self.bboxes.get(sample["id"], {})
        try:
            depth, k = self._load(sample)
        except (OSError, ValueError, KeyError):
            return ToolResult(name, None, status="depth_missing")
        try:
            if name == TOOL_OBJ_OBJ_DIST:
                a = self._bbox(entry.get("bboxA_px") or entry.get("bbox_px"), depth.shape)
                b = self._bbox(entry.get("bboxB_px"), depth.shape)
                if a is None or b is None:
                    return ToolResult(name, None, status="target_not_found")
                za, zb = bbox_depth_median(depth, a), bbox_depth_median(depth, b)
                if za <= .5 or zb <= .5:
                    return ToolResult(name, None, status="bad_geometry")
                pa = back_project(.5 * (a[0] + a[2]), .5 * (a[1] + a[3]), za, k)
                pb = back_project(.5 * (b[0] + b[2]), .5 * (b[1] + b[3]), zb, k)
                value = float(np.linalg.norm(pa - pb))
            else:
                bbox = self._bbox(entry.get("bbox_px"), depth.shape)
                if bbox is None:
                    return ToolResult(name, None, status="target_not_found")
                z = bbox_depth_median(depth, bbox)
                if z <= .5:
                    return ToolResult(name, None, status="bad_geometry")
                if name == TOOL_EGO_DIST:
                    value = float(np.linalg.norm(back_project(.5 * (bbox[0] + bbox[2]), .5 * (bbox[1] + bbox[3]), z, k)))
                else:
                    value = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * z / k[0, 0]
        except (ValueError, TypeError, OverflowError):
            return ToolResult(name, None, status="bad_geometry")
        if not np.isfinite(value) or value <= 0:
            return ToolResult(name, None, status="bad_geometry")
        return ToolResult(name, float(value))

    def check_assets(self, rows):
        return dict(Counter(self.call(row, ToolCall(TASK_TO_TOOL[row["task"]])).status for row in rows))
