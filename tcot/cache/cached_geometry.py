from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import clock_geometry, lateral_geometry, vertical_geometry
from .metric_geometry import (
    MIN_MASK_PIXELS, backproject_mask_with_valid, compute_physical_value,
    filter_outliers, load_mask_from_npz, rescale_K, resize_depth_to, resize_mask_to,
)
from .. import o3dvqa_protocol as protocol
from ..tool_schema import (
    O3DVQA_TOOL_TO_QNAME, TOOL_EGO_DIST, TOOL_OBJ_OBJ_DIST, TOOL_OBJ_SIZE,
    ToolResult, quantity_description_for_o3dvqa_question, unit_for_o3dvqa_question,
)

METRIC_QUESTIONS = set(O3DVQA_TOOL_TO_QNAME.values())
PAIR_TASKS = {"distance_data", "horizontal_distance_data", "vertical_distance_data"}
TARGET_HW = (480, 640)


class CachedGeometryBackend:
    """Historical predicted-asset protocol; no historical answers are loaded."""

    def __init__(self, manifest, relation_cues=None):
        self.path = Path(manifest).resolve()
        self.entries = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.entries, dict):
            raise ValueError("Perception manifest must be an ID-to-assets mapping")
        self.relation_path = Path(relation_cues).resolve() if relation_cues else None
        self.cues = {}
        if self.relation_path is not None:
            from external_tools.o3dvqa_extraction_common import iter_items

            for _, _, row in iter_items(json.loads(self.relation_path.read_text(encoding="utf-8"))):
                sid = str(row["id"])
                if sid in self.cues:
                    raise ValueError(f"Duplicate relation cue ID: {sid}")
                self.cues[sid] = row
        self._verified = {}

    def resolve(self, name):
        if not isinstance(name, (str, Path)):
            raise ValueError("Invalid cached asset path")
        path = Path(name)
        return path if path.is_absolute() else self.path.parent / path

    def verify_asset(self, path, expected):
        if expected is None:
            return
        from dronepur_common import file_digest

        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._verified:
            self._verified[key] = file_digest(path)
        if self._verified[key] != expected:
            raise ValueError(f"cached_asset_fingerprint_mismatch: {path}")

    def load(self, sample, call=None, metric=False, roles=None):
        entry = self.entries.get(str(sample["id"]))
        if not isinstance(entry, dict):
            raise ValueError("missing_perception_assets")
        if entry.get("uses_gt") is True:
            raise ValueError("GT assets are not permitted")
        if metric and entry.get("depth_unit") != "m":
            raise ValueError("Metric tools require metric depth in meters")
        source = entry.get("source") or {}
        objects = entry.get("objects") or {}
        if not isinstance(source, dict) or not isinstance(objects, dict):
            raise ValueError("Invalid asset manifest schema")
        if entry.get("image_name") and entry["image_name"] != sample.get("image_name"):
            raise ValueError("cached_image_mismatch")
        depth_path = self.resolve(entry["depth_path"])
        self.verify_asset(depth_path, source.get("depth_sha256"))
        if source.get("rgb_sha256"):
            self.verify_asset(sample["rgb_path"], source["rgb_sha256"])
        with np.load(depth_path, allow_pickle=False) as data:
            if "cache_config" in data:
                settings = json.loads(str(data["cache_config"].item()))
                if not isinstance(settings, dict) or settings.get("uses_gt") is True:
                    raise ValueError("Invalid predicted depth metadata")
                if settings.get("depth_unit") and settings["depth_unit"] != entry.get("depth_unit"):
                    raise ValueError("cached_depth_unit_mismatch")
                if source.get("depth_backend") and settings.get("depth_backend") and source["depth_backend"] != settings["depth_backend"]:
                    raise ValueError("cached_depth_backend_mismatch")
            depth = data["depth"].astype(np.float32)
            k = data["K"].astype(np.float64) if "K" in data else data["intrinsics"].astype(np.float64)
            valid = data["valid_mask"].astype(bool) if "valid_mask" in data else np.ones(depth.shape, bool)
        if depth.ndim != 2 or valid.shape != depth.shape:
            raise ValueError("Invalid depth canvas")
        if entry.get("intrinsics") is not None:
            k = np.asarray(entry["intrinsics"], dtype=np.float64)
        if entry.get("intrinsics_path"):
            k = np.asarray(np.load(self.resolve(entry["intrinsics_path"]), allow_pickle=False), dtype=np.float64)
        if roles is None:
            roles = list(objects)
        masks, prompts, paths = {}, {}, {}
        for role in roles:
            obj = objects.get(role)
            if not isinstance(obj, dict) or not isinstance(obj.get("prompt"), str) or not obj.get("mask_path"):
                raise ValueError("missing_predicted_mask")
            path = self.resolve(obj["mask_path"])
            self.verify_asset(path, (source.get("mask_sha256") or {}).get(role))
            mask = load_mask_from_npz(str(path))
            if mask is None:
                raise ValueError("missing_predicted_mask")
            masks[role] = mask if metric else resize_mask_to(mask, depth.shape)
            paths[role], prompts[role] = path, obj["prompt"]
        valid &= np.isfinite(depth) & (depth > .5) & (depth < 65000.)
        return depth, k, valid, masks, prompts, paths

    def metric(self, sample, call):
        qname = (sample.get("qa_info") or {}).get("question_name")
        task = O3DVQA_TOOL_TO_QNAME.get(call.name)
        if task is None:
            task = {TOOL_EGO_DIST: "distance2agent", TOOL_OBJ_OBJ_DIST: "distance_data", TOOL_OBJ_SIZE: "width_data"}.get(call.name)
        unit = unit_for_o3dvqa_question(task or qname)
        quantity = quantity_description_for_o3dvqa_question(task or qname)
        if task is None:
            return ToolResult(call.name, None, unit=unit, quantity=quantity, status="invalid_call")
        roles = ["subject", "reference"] if task in PAIR_TASKS else ["subject"]
        try:
            depth, k, _, masks, _, _ = self.load(sample, metric=True, roles=roles)
        except (OSError, ValueError, KeyError):
            return ToolResult(call.name, None, unit=unit, quantity=quantity, status="target_not_found" if str(sample["id"]) not in self.entries else "depth_missing")
        entry = self.entries[str(sample["id"])]
        backend = (entry.get("source") or {}).get("depth_backend")
        native_hw = (378, 504) if backend == "da3" else (480, 640)
        depth = resize_depth_to(depth, TARGET_HW)
        k = rescale_K(k, native_hw[1], native_hw[0], TARGET_HW[1], TARGET_HW[0])
        points = {}
        for role in roles:
            mask = resize_mask_to(masks[role], TARGET_HW)
            cloud, _ = backproject_mask_with_valid(mask, depth, k)
            if len(cloud) < MIN_MASK_PIXELS:
                return ToolResult(call.name, None, unit=unit, quantity=quantity, status="bad_geometry")
            points[role] = filter_outliers(cloud)
        try:
            value = compute_physical_value(task, points["subject"], points.get("reference"))
        except (ValueError, TypeError):
            value = None
        if value is None or not np.isfinite(value) or value <= 0:
            return ToolResult(call.name, None, unit=unit, quantity=quantity, status="bad_geometry")
        return ToolResult(call.name, float(value), unit=unit, quantity=quantity)

    def relation(self, sample, call, image=None):
        cue = self.cues.get(str(sample["id"]))
        decision = protocol.validate_relation_binding(sample, cue, call.target, call.target_b)
        if not decision.usable:
            return dict(status="failed", reason=decision.reason), image
        decision = protocol.validate_relation_evidence(cue, (sample.get("qa_info") or {}).get("question_name"), max_cues=6)
        return decision.evidence if decision.usable else dict(status="failed", reason=decision.reason), image

    def stage_specs(self, sample):
        cue = self.cues.get(str(sample["id"]), {})
        prompts = {role: obj.get("prompt", "") for role, obj in (cue.get("sam_prompts") or {}).items() if isinstance(obj, dict)}
        question = sample["query_question"]
        vertical = vertical_geometry.vertical_question(question, prompts) and all(
            (cue.get("sam_prompts") or {}).get(role, {}).get("mask_path")
            and (cue.get("sam_prompts") or {}).get(role, {}).get("num_masks", 0) > 0
            for role in ("subject", "reference")
        )
        lateral = None if vertical else lateral_geometry.question_spec(question, prompts)
        qname = (sample.get("qa_info") or {}).get("question_name")
        clock = clock_geometry.question_spec(question, prompts) if qname == "direction_data" else None
        return dict(vertical=bool(vertical), lateral=lateral, clock=clock, prompts=prompts)

    def stage_evidence(self, sample, stage, spec=None):
        roles = ["subject", "reference"] if stage == "vertical" else spec["required_roles"]
        if spec is not None and not spec["applicable"]:
            return dict(status="failed", reason=spec["reason"])
        try:
            depth, k, valid, masks, prompts, paths = self.load(sample, roles=roles)
            cue = self.cues[str(sample["id"])]
            stage_prompts = {role: obj.get("prompt", "") for role, obj in (cue.get("sam_prompts") or {}).items() if isinstance(obj, dict)}
            if any(prompts[role] != stage_prompts.get(role) for role in roles):
                raise ValueError("Cue prompt and predicted mask identity differ")
            if stage == "vertical":
                return vertical_geometry.build_evidence(depth, k, valid, masks, stage_prompts)
            module = lateral_geometry if stage == "lateral" else clock_geometry
            ev = module.build_evidence(depth, k, valid, masks, stage_prompts, spec)
            if stage == "lateral":
                return dict(question_spec=spec, lateral3d_evidence=ev,
                            source=dict(masks={role: dict(path=str(paths[role])) for role in roles}))
            return dict(question_spec=spec, clock3d_evidence=ev)
        except (OSError, ValueError, KeyError) as error:
            return dict(status="failed", reason=str(error))
