"""Tool requests and results for Qwen3-VL TCoT on Open3DVQA."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

TOOL_EGO_DIST = "ego_distance_computation"
TOOL_OBJ_OBJ_DIST = "object_to_object_distance_computation"
TOOL_OBJ_SIZE = "object_size_computation"

TOOL_O3D_DISTANCE2AGENT = "distance_to_camera_computation"
TOOL_O3D_VERTICAL_DISTANCE2AGENT = "vertical_distance_to_camera_computation"
TOOL_O3D_HORIZONTAL_DISTANCE2AGENT = "horizontal_distance_to_camera_computation"
TOOL_O3D_DISTANCE_DATA = "distance_between_objects_computation"
TOOL_O3D_VERTICAL_DISTANCE_DATA = "vertical_distance_between_objects_computation"
TOOL_O3D_HORIZONTAL_DISTANCE_DATA = "horizontal_distance_between_objects_computation"
TOOL_O3D_WIDTH_DATA = "object_width_computation"
TOOL_O3D_HEIGHT_DATA = "object_height_computation"
TOOL_O3D_VOLUME_DATA = "object_volume_computation"
TOOL_O3D_RELATION_EVIDENCE = "relation_evidence"

O3DVQA_QNAME_TO_TOOL = {
    "distance2agent": TOOL_O3D_DISTANCE2AGENT,
    "vertical_distance2agent": TOOL_O3D_VERTICAL_DISTANCE2AGENT,
    "horizontal_distance2agent": TOOL_O3D_HORIZONTAL_DISTANCE2AGENT,
    "distance_data": TOOL_O3D_DISTANCE_DATA,
    "vertical_distance_data": TOOL_O3D_VERTICAL_DISTANCE_DATA,
    "horizontal_distance_data": TOOL_O3D_HORIZONTAL_DISTANCE_DATA,
    "width_data": TOOL_O3D_WIDTH_DATA,
    "height_data": TOOL_O3D_HEIGHT_DATA,
    "volume_data": TOOL_O3D_VOLUME_DATA,
}
O3DVQA_TOOL_TO_QNAME = {value: key for key, value in O3DVQA_QNAME_TO_TOOL.items()}
O3DVQA_QNAME_TO_DESCRIPTION = {
    "distance2agent": "full 3D distance from the camera to the target object center",
    "vertical_distance2agent": "vertical camera-frame offset from the camera to the target object center",
    "horizontal_distance2agent": "horizontal/lateral camera-frame offset from the camera to the target object center",
    "distance_data": "full 3D distance between the two object centers",
    "vertical_distance_data": "vertical camera-frame separation between the two object centers",
    "horizontal_distance_data": "horizontal/lateral camera-frame separation between the two object centers",
    "width_data": "camera-frame left-right width of the target object",
    "height_data": "camera-frame vertical height of the target object",
    "volume_data": "camera-frame 3D AABB volume of the target object",
}
O3DVQA_TWO_TARGET_TOOLS = {TOOL_O3D_DISTANCE_DATA, TOOL_O3D_VERTICAL_DISTANCE_DATA, TOOL_O3D_HORIZONTAL_DISTANCE_DATA}
TWO_TARGET_TOOL_NAMES = O3DVQA_TWO_TARGET_TOOLS | {TOOL_OBJ_OBJ_DIST}
VALID_TOOL_NAMES = {*O3DVQA_TOOL_TO_QNAME, TOOL_O3D_RELATION_EVIDENCE, TOOL_EGO_DIST, TOOL_OBJ_OBJ_DIST, TOOL_OBJ_SIZE}


def quantity_description_for_o3dvqa_question(question_name):
    return O3DVQA_QNAME_TO_DESCRIPTION.get(question_name, "the requested metric quantity")


def unit_for_o3dvqa_question(question_name):
    return "m^3" if question_name == "volume_data" else "m"


O3DVQA_TOOL_MENU_BLOCK = (
    "Available external tools (choose exactly one):\n"
    "  1. distance_to_camera_computation(target)\n"
    "     -> Full 3D distance from the camera to `target`.\n"
    "  2. vertical_distance_to_camera_computation(target)\n"
    "     -> Vertical component of the camera-to-target offset.\n"
    "  3. horizontal_distance_to_camera_computation(target)\n"
    "     -> Horizontal/lateral component of the camera-to-target offset.\n"
    "  4. distance_between_objects_computation(target_a, target_b)\n"
    "     -> Full 3D distance between two objects.\n"
    "  5. vertical_distance_between_objects_computation(target_a, target_b)\n"
    "     -> Vertical component of the offset between two objects.\n"
    "  6. horizontal_distance_between_objects_computation(target_a, target_b)\n"
    "     -> Horizontal/lateral component of the offset between two objects.\n"
    "  7. object_width_computation(target)\n"
    "     -> Left-right width of `target` in the camera frame.\n"
    "  8. object_height_computation(target)\n"
    "     -> Vertical height of `target` in the camera frame.\n"
    "  9. object_volume_computation(target)\n"
    "     -> 3D volumetric extent of `target`.\n"
    "  10. relation_evidence(target, reference)\n"
    "     -> SAM3-grounded object geometry from estimated depth for qualitative "
    "spatial relations. Reports normalized image centers, bounding boxes, mean "
    "depths, valid-depth ratios, and task-aligned measurements when available; "
    "infer the final relation or direction yourself. Use `reference` for "
    "object-to-object questions; omit it for object-to-viewer/agent questions.\n"
    "\n"
    "Emit exactly one function call inside a <tool_call> ... </tool_call> "
    "block. The payload must be a single JSON object. Examples:\n"
    "  <tool_call>{\"name\": \"distance_to_camera_computation\", "
    "\"target\": \"the red car in the middle\"}</tool_call>\n"
    "  <tool_call>{\"name\": \"distance_between_objects_computation\", "
    "\"target_a\": \"...\", \"target_b\": \"...\"}</tool_call>\n"
    "  <tool_call>{\"name\": \"relation_evidence\", "
    "\"target\": \"the pavilion\", \"reference\": \"the wall\"}</tool_call>\n"
    "For single-target metric functions use `target`; for two-target metric "
    "functions use `target_a` and `target_b`; for relation_evidence use "
    "`target` and optional `reference`. Do not output anything else in this turn."
)


@dataclass
class ToolCall:
    name: str
    target: Optional[str] = None
    target_a: Optional[str] = None
    target_b: Optional[str] = None
    reference: Optional[str] = None

    @property
    def is_two_target(self):
        return self.name in TWO_TARGET_TOOL_NAMES


@dataclass
class ToolResult:
    name: str
    value: Optional[float]
    unit: str = "m"
    status: str = "ok"
    quantity: Optional[str] = None

    def to_prompt_json(self):
        payload = dict(name=self.name, value=None, unit=self.unit, status=self.status)
        if self.quantity is not None:
            payload["quantity"] = self.quantity
        if self.status == "ok" and self.value is not None:
            payload["value"] = round(float(self.value), 4)
        return json.dumps(payload, ensure_ascii=False)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_tool_call(text: str) -> Tuple[Optional[ToolCall], str]:
    """Try to extract a single tool call from the model's Turn-2 output.

    Returns ``(ToolCall or None, parse_status)`` where ``parse_status`` is
    one of {``"ok"``, ``"no_call"``, ``"bad_json"``, ``"unknown_name"``,
    ``"missing_target"``}. The first successful <tool_call> block wins.
    If no fenced block is found, we try to salvage the first JSON object
    that mentions one of the valid tool names -- this is a small robustness
    concession because the model sometimes forgets the fence.
    """
    if not text:
        return None, "no_call"

    candidates = _TOOL_CALL_RE.findall(text)
    if not candidates:
        # Fallback: any {...} object that mentions one of the valid tool names.
        for m in _JSON_OBJ_RE.findall(text):
            if any(name in m for name in VALID_TOOL_NAMES):
                candidates = [m]
                break
    if not candidates:
        return None, "no_call"

    saw_unknown_name = False
    for payload in candidates:
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) or name not in VALID_TOOL_NAMES:
            saw_unknown_name = True
            continue
        if name == TOOL_O3D_RELATION_EVIDENCE:
            for aliases in (("target", "subject", "target_a"), ("reference", "target_b", "object")):
                values = [obj[key] for key in aliases if obj.get(key) is not None]
                if any(not isinstance(value, str) for value in values):
                    return None, "invalid_target"
                phrases = {value.strip().casefold() for value in values if value.strip()}
                if len(phrases) > 1:
                    return None, "conflicting_targets"
            target = obj.get("target") or obj.get("subject") or obj.get("target_a")
            reference = obj.get("reference") or obj.get("target_b") or obj.get("object")
            if not target:
                return None, "missing_target"
            return ToolCall(
                name=name,
                target=str(target).strip(),
                target_a=str(target).strip(),
                target_b=str(reference).strip() if reference else None,
            ), "ok"
        if name in TWO_TARGET_TOOL_NAMES:
            ta = obj.get("target_a") or obj.get("targetA")
            tb = obj.get("target_b") or obj.get("targetB")
            if not ta or not tb:
                return None, "missing_target"
            return ToolCall(name=name, target_a=str(ta).strip(),
                            target_b=str(tb).strip()), "ok"
        target = obj.get("target")
        if not target:
            return None, "missing_target"
        return ToolCall(name=name, target=str(target).strip()), "ok"

    return None, "unknown_name" if saw_unknown_name else "bad_json"
