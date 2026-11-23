from __future__ import annotations
import json, math, re, unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from .question_prompts import _strip_letter_only_tail, build_mcq_reasoning_system_prompt, build_mcq_reasoning_prompt
from .tool_schema import TOOL_O3D_RELATION_EVIDENCE, ToolResult


O3DVQA_TOOL_MENU_BLOCK = (
    "Available geometry functions (choose exactly one):\n"
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
    "\n"
    "Emit exactly one function call inside a <tool_call> ... </tool_call> "
    "block. The payload must be a single JSON object. Examples:\n"
    "  <tool_call>{\"name\": \"distance_to_camera_computation\", "
    "\"target\": \"the red car in the middle\"}</tool_call>\n"
    "  <tool_call>{\"name\": \"distance_between_objects_computation\", "
    "\"target_a\": \"...\", \"target_b\": \"...\"}</tool_call>\n"
    "For single-target functions use `target`; for two-target functions "
    "use `target_a` and `target_b`. Do not output anything else in this turn."
)


METRIC_QNAMES = {
    "distance2agent",
    "vertical_distance2agent",
    "horizontal_distance2agent",
    "distance_data",
    "vertical_distance_data",
    "horizontal_distance_data",
    "width_data",
    "height_data",
    "volume_data",
}

@dataclass
class EvidenceDecision:
    usable: bool
    status: str
    reason: str
    evidence: Optional[Dict[str, Any]] = None
    prompt_text: Optional[str] = None

def is_metric_question_name(question_name: Optional[str]) -> bool:
    return bool(question_name in METRIC_QNAMES)

def _normalize_object_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words)

def _question_relation_targets(question: str, question_name: Optional[str]) -> Optional[Dict[str, str]]:
    if question_name == "direction_data":
        match = re.fullmatch(r"\s*If you are at (.+),\s*where will you find (.+?)\?\s*", question, re.I)
        if match:
            return {"subject": match[2], "reference": match[1]}
    if question_name == "direction2agent":
        match = re.search(r"\b(?:direction|angle) of (.+?)[?.]?\s*$", question, re.I)
        if match:
            return {"subject": match[1]}
    return None

def _has_compact_target(target: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(target, dict) or not isinstance(target.get("prompt"), str):
        return False
    if not target["prompt"].strip():
        return False
    center = target.get("center_xy_normalized")
    bbox = target.get("bbox_xyxy_normalized")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return False
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    values = [*center, *bbox, target.get("depth_mean")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return False
    if not all(0 <= v <= 1 for v in [*center, *bbox]) or target["depth_mean"] <= 0:
        return False
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return False
    ratio = target.get("valid_depth_ratio")
    return ratio is None or (isinstance(ratio, (int, float)) and math.isfinite(ratio) and 0 < ratio <= 1)

def _needs_reference(question_name: Optional[str], evidence: Dict[str, Any]) -> bool:
    objects = evidence.get("objects") or evidence.get("grounding") or {}
    if objects.get("reference") is not None:
        return True
    if not question_name:
        return False
    if "2agent" in question_name:
        return False
    if question_name in {"direction2agent"}:
        return False
    return True

def validate_relation_binding(
    item: Dict[str, Any],
    cue_item: Optional[Dict[str, Any]],
    requested_subject: Optional[str],
    requested_reference: Optional[str],
) -> EvidenceDecision:
    if not cue_item:
        return EvidenceDecision(False, "missing", "missing_relation_cue")
    question = item.get("query_question") or item.get("question") or ""
    qname = (item.get("qa_info") or {}).get("question_name")
    if str(cue_item.get("id")) != str(item.get("id")):
        return EvidenceDecision(False, "mismatched_cue", "cue_id_mismatch")
    if _normalize_object_text(cue_item.get("question")) != _normalize_object_text(question):
        return EvidenceDecision(False, "mismatched_cue", "cue_question_mismatch")
    if item.get("image_name") and cue_item.get("image_name") != item["image_name"]:
        return EvidenceDecision(False, "mismatched_cue", "cue_image_mismatch")
    if (cue_item.get("qa_info") or {}).get("question_name") != qname:
        return EvidenceDecision(False, "mismatched_cue", "cue_question_type_mismatch")
    targets = (cue_item.get("target_extraction") or {}).get("targets") or {}
    expected = _question_relation_targets(question, qname) or targets
    reference_needed = bool(qname and "2agent" not in qname)
    if not qname:
        reference_needed = bool(expected.get("reference")) or bool(requested_reference)
    elif "2agent" in qname and (targets.get("reference") or requested_reference):
        return EvidenceDecision(False, "mismatched_cue", "unexpected_agent_reference")
    roles = ("subject", "reference") if reference_needed else ("subject",)
    if not qname and not reference_needed:
        return EvidenceDecision(False, "incomplete", "unknown_object_count")
    evidence = cue_item.get("relation_evidence") or {}
    objects = evidence.get("objects") or evidence.get("grounding") or {}
    requested = {"subject": requested_subject, "reference": requested_reference}
    normalized_question = " " + _normalize_object_text(question) + " "
    for role in roles:
        phrase = _normalize_object_text(expected.get(role))
        if not phrase:
            return EvidenceDecision(False, "incomplete", f"missing_expected_{role}")
        if " " + phrase + " " not in normalized_question:
            return EvidenceDecision(False, "mismatched_cue", f"expected_{role}_not_in_question")
        if _normalize_object_text(targets.get(role)) != phrase:
            return EvidenceDecision(False, "mismatched_cue", f"extracted_{role}_mismatch")
        if _normalize_object_text(requested[role]) != phrase:
            return EvidenceDecision(False, "mismatched_cue", f"requested_{role}_mismatch")
        obj = objects.get(role)
        if not _has_compact_target(obj):
            return EvidenceDecision(False, "incomplete", f"missing_or_incomplete_{role}")
        if _normalize_object_text(obj.get("prompt")) != phrase or obj.get("role", role) != role:
            return EvidenceDecision(False, "mismatched_cue", f"cached_{role}_mismatch")
    if reference_needed and _normalize_object_text(expected.get("subject")) == _normalize_object_text(expected.get("reference")):
        return EvidenceDecision(False, "ambiguous", "indistinguishable_object_roles")
    if not reference_needed and objects.get("reference") is not None:
        return EvidenceDecision(False, "mismatched_cue", "unexpected_reference")
    return EvidenceDecision(True, "ok", "object_roles_verified")

def validate_relation_evidence(
    cue_item: Optional[Dict[str, Any]],
    question_name: Optional[str] = None,
    max_cues: int = 6,
) -> EvidenceDecision:
    if not cue_item:
        return EvidenceDecision(False, "missing", "missing_relation_cue")

    evidence = cue_item.get("relation_evidence") or {}
    status = str(evidence.get("status") or cue_item.get("status") or "")
    if status != "ok":
        return EvidenceDecision(False, status or "failed", f"relation_status={status or 'missing'}")

    source = evidence.get("source") or cue_item.get("source") or {}
    if source.get("uses_gt") is True:
        return EvidenceDecision(False, "invalid_source", "relation_cue_uses_gt")
    if source.get("mask_source") and source.get("mask_source") != "sam3":
        return EvidenceDecision(False, "invalid_source", "mask_source_not_sam3")

    objects = evidence.get("objects") or evidence.get("grounding") or {}
    subject_ok = _has_compact_target(objects.get("subject"))
    if not subject_ok:
        return EvidenceDecision(False, "incomplete", "missing_or_incomplete_subject")

    reference_needed = _needs_reference(question_name, evidence)
    reference_ok = _has_compact_target(objects.get("reference"))
    if reference_needed and not reference_ok:
        return EvidenceDecision(False, "incomplete", "missing_or_incomplete_reference")

    prompt_cues = []
    for cue in evidence.get("prompt_cues") or cue_item.get("prompt_cues") or []:
        text = str(cue).strip()
        if not text:
            continue
        if any(
            bad in text
            for bad in (
                "visible_size",
                "visible image area",
                " left of ",
                " right of ",
                " above ",
                " below ",
                " closer than ",
                " farther than ",
            )
        ):
            continue
        if not text.startswith("- "):
            text = f"- {text}"
        prompt_cues.append(text)
        if max_cues is not None and len(prompt_cues) >= max_cues:
            break

    if not prompt_cues:
        for role in ("subject", "reference"):
            obj = objects.get(role)
            if not isinstance(obj, dict):
                continue
            pieces = [f"{role}='{obj.get('prompt') or role}'"]
            if obj.get("center_xy_normalized") is not None:
                pieces.append(f"center_xy_normalized={obj.get('center_xy_normalized')}")
            if obj.get("bbox_xyxy_normalized") is not None:
                pieces.append(f"bbox_xyxy_normalized={obj.get('bbox_xyxy_normalized')}")
            if obj.get("depth_mean") is not None:
                pieces.append(f"depth_mean={obj.get('depth_mean')}")
            if obj.get("valid_depth_ratio") is not None:
                pieces.append(f"valid_depth_ratio={obj.get('valid_depth_ratio')}")
            prompt_cues.append("- " + ", ".join(pieces) + ".")

    if not prompt_cues:
        return EvidenceDecision(False, "incomplete", "no_object_geometry_cues")

    payload = {
        "name": TOOL_O3D_RELATION_EVIDENCE,
        "status": "ok",
        "evidence_type": "object_geometry",
        "objects": objects,
        "prompt_cues": prompt_cues,
        "source": source,
    }
    prompt_text = "\n".join(prompt_cues)
    return EvidenceDecision(True, "ok", "ok", payload, prompt_text)

UNIVERSAL_O3DVQA_TOOL_MENU_BLOCK = (
    "Available external tools (choose exactly one):\n"
    + O3DVQA_TOOL_MENU_BLOCK.replace(
        "Available geometry functions (choose exactly one):\n", ""
    ).replace(
        "Do not output anything else in this turn.",
        ""
    )
    + "  10. relation_evidence(target, reference)\n"
    "     -> Returns SAM3-grounded object geometry measurements from estimated "
    "depth for qualitative spatial-relation questions. The tool reports each "
    "object's normalized image center, normalized bounding box, mean depth, "
    "and valid-depth ratio; you must infer the "
    "left/right/above/below/nearer/farther relation yourself. Use `reference` "
    "when the question compares two objects; omit it for object-to-viewer/agent "
    "questions.\n"
    "\n"
    "Emit exactly one function call inside a <tool_call> ... </tool_call> block. "
    "The payload must be a single JSON object. Examples:\n"
    "  <tool_call>{\"name\": \"relation_evidence\", \"target\": "
    "\"the pavilion\", \"reference\": \"the wall\"}</tool_call>\n"
    "  <tool_call>{\"name\": \"distance_to_camera_computation\", "
    "\"target\": \"the red car\"}</tool_call>\n"
    "For single-target metric functions use `target`; for two-target metric "
    "functions use `target_a` and `target_b`; for relation_evidence use "
    "`target` and optional `reference`. Do not output anything else in this turn."
)

_UNIVERSAL_SYS = (
    "You are a spatial-reasoning assistant answering questions about outdoor "
    "urban imagery. This conversation has two turns: (1) request exactly one "
    "external tool result via a <tool_call> block and do NOT answer yet; "
    "(2) answer the original question using the image and any valid external "
    "evidence. Treat tool output as noisy reference evidence, not ground truth."
)

def build_universal_tool_first_system_prompt() -> str:
    return _UNIVERSAL_SYS

def build_turn1_universal_tool_call_o3dvqa(question_text: str) -> str:
    q = _strip_letter_only_tail(question_text or "")
    return (
        f"Question:\n{q}\n\n"
        "First, request one external tool result to help answer the question. "
        "Use a numeric geometry function for metric distance/size/volume "
        "questions. Use relation_evidence for qualitative spatial relation, "
        "direction, left/right/above/below/front/behind, true/false, or "
        "object-choice relation questions; it returns object geometry only, "
        "not the final relation.\n\n"
        + UNIVERSAL_O3DVQA_TOOL_MENU_BLOCK
        + "\n\n### RESPONSE FORMAT (MANDATORY):\n"
        "Emit exactly one <tool_call>...</tool_call> block. Do NOT write any "
        "reasoning, do NOT write A/B/C/D, and do NOT answer the question in "
        "this turn."
    )

def build_metric_tool_fusion_prompt(
    question_text: str,
    task_type: Optional[str],
    tool_result: ToolResult,
) -> str:
    payload = tool_result.to_prompt_json()
    header = (
        "A numeric geometry tool has been called for you and returned:\n"
        f"<tool_response>{payload}</tool_response>\n"
        "If a `quantity` field is present, it states exactly what was measured. "
        "This measurement may be noisy or wrong; treat it as one piece of "
        "evidence, not ground truth.\n\n"
        "Now answer by anchor-based reasoning. Run Step 1, Step 2 and Step 3 "
        "independently from the tool value, so you have your own estimate. "
        "In Step 4, compare your estimate with the tool measurement and use "
        "the tool only as a soft reference.\n\n"
    )
    return header + build_mcq_reasoning_prompt(question_text, task_type)

CLOCK_DIRECTION_QNAMES = {"direction2agent", "direction_data"}

def infer_direction_roles_from_question(question_text: str) -> Optional[Dict[str, str]]:
    question = (question_text or "").strip()
    pair = re.fullmatch(
        r"If you are at (.+),\s*where will you find (.+?)\?\s*", question, re.I,
    )
    if pair:
        return {"setting": "object_relative", "reference": pair[1].strip(), "subject": pair[2].strip()}
    relative = re.search(
        r"\b(?:direction|angle)\s+of\s+(.+?)\s+(?:relative to|from)\s+(.+?)\s*[?.]?\s*$",
        question, re.I,
    )
    if relative:
        reference = relative[2].strip()
        is_viewer = _normalize_object_text(reference) in {"you", "me", "camera", "viewer", "agent"}
        return {
            "setting": "ego" if is_viewer else "object_relative",
            "reference": "viewer/agent" if is_viewer else reference,
            "subject": relative[1].strip(),
        }
    ego = re.search(r"\b(?:direction|angle)\s+of\s+(.+?)\s*[?.]?\s*$", question, re.I)
    if ego:
        subject = ego[1].strip()
        if re.search(r"\b(?:relative|from|compared|with respect)\b", subject, re.I):
            return None
        return {"setting": "ego", "reference": "viewer/agent", "subject": subject}
    return None

THREE_STAGE_FUSION_SYSTEM_V2 = (
    "Answer the original spatial question using its image, an independent draft, "
    "and external object evidence. Verify object identity, reference frame, "
    "comparison direction and answer type before changing the draft. Tool success "
    "does not guarantee correct object binding or a reason to change an answer. "
    "A choice question requires one selected object, never a Yes/No assertion. "
    "Follow the explicit final-output contract and output only the final answer."
)

def relation_answer_contract(question_text: str, question_name: Optional[str]) -> tuple[str, str]:
    qname = question_name or ""
    if qname in CLOCK_DIRECTION_QNAMES:
        return "clock_direction", (
            "Return exactly one integer clock position from 1 to 12, such as \"5 o'clock\". "
            "Do not output degrees, coordinates, two directions, or a Yes/No answer."
        )
    if qname.endswith(("_choice", "_multichoice")) or re.match(
        r"\s*(?:which\b|who\b|what\s+object\b)", question_text, re.I,
    ):
        return "object_choice", (
            "This is an OBJECT-CHOICE question, NOT a yes/no question. Return the full "
            "description of exactly one selected object from the question; include its "
            "distinguishing attributes. If the question supplies Same/Unknown options, "
            "select that option only when justified. Do not return a bare option letter. "
            "Never begin with Yes or No, and never replace the choice with a negative "
            "statement about the tool subject. If subject does not satisfy the requested "
            "comparison, explicitly select the reference when the comparison supports it."
        )
    if qname.endswith(("_predicate", "_tfqa", "_relationship2agent")) or re.match(
        r"\s*(?:is|are|does|do|can|could|would)\b", question_text, re.I,
    ):
        return "boolean", (
            "This is a BOOLEAN question. Return Yes or No for the exact assertion in "
            "the question, optionally followed by one short sentence naming the same "
            "subject and reference. Do not answer by selecting a different object or "
            "giving a bare option letter."
        )
    return "question_defined", (
        "Determine the answer type from the ORIGINAL question: select one explicitly "
        "named object for a choice, or give Yes/No only for a boolean assertion. Do "
        "not use the tool's subject/reference labels to change the question type."
    )

def build_three_stage_fusion_prompt_v2_1(
    question_text: str, draft: str, evidence_kind: str, evidence: Dict[str, Any],
    question_name: Optional[str],
) -> str:
    if evidence_kind != "relation" or evidence.get("status") != "ok" or is_metric_question_name(question_name):
        raise ValueError("Three-stage v2.1 is restricted to valid relation evidence")
    answer_type, contract = relation_answer_contract(question_text, question_name)
    if answer_type == "boolean":
        contract = (
            "If the conclusion must change, answer Yes or No for the exact original "
            "assertion. Do not add a new explanation or reverse the subject/reference."
        )
    elif answer_type == "clock_direction":
        contract = (
            "Give one clock direction from 1 to 12 for the target relative to the "
            "requested reference. Preserve the draft's target/reference description "
            "and sentence form; if a correction is justified, change only the clock "
            "direction and any wording strictly required for consistency. Do not "
            "turn a grounded sentence into a bare clock number."
        )
    relation = (question_name or "").split("_", 1)[0]
    if relation not in {"left", "right", "above", "below", "front", "behind", "big", "small", "wide", "thin", "tall", "short"}:
        match = re.search(r"\b(left|right|above|below|front|behind)\b", question_text, re.I)
        relation = match[1].lower() if match else "question-defined"
    if answer_type == "clock_direction":
        roles = infer_direction_roles_from_question(question_text)
        geometry = (
            "For 'If you are at A, where will you find B?', reference=A and target=B. "
            "Do not substitute the camera as origin. A camera-depth mean and image "
            "center do not determine a world-plane relative bearing or object facing. "
            "Do not reinterpret image y as front/back or invent pose/calibration. "
            "If the supplied evidence cannot resolve the requested bearing, copy "
            "the independent answer unchanged."
        )
        if roles:
            geometry += " Question-derived roles: " + json.dumps(roles, ensure_ascii=False) + "."
    elif relation in {"above", "below"}:
        geometry = (
            "Above/below may refer to physical 3D height, not vertical position in "
            "an aerial image. Image center y and camera-depth mean alone do not "
            "establish gravity-aligned height. Never use image y order to overturn "
            "a physical above/below judgment. Only if the original question explicitly "
            "asks about image/screen position, smaller image y means above and larger "
            "image y means below. Otherwise these measurements support object "
            "localization only: copy the independent answer unchanged when no "
            "applicable height evidence is supplied."
        )
    elif relation in {"left", "right"}:
        geometry = (
            "For viewer/image-relative left/right, center_xy_normalized=[x,y] and "
            "x increases to the right: subject left of reference means x_subject "
            "< x_reference, and right means x_subject > x_reference. A left choice "
            "selects the smaller x and a right choice the larger x, with the actual "
            "object identities preserved. Depth does not decide image left/right. "
            "Do not apply this rule to an object's facing-relative or other physical "
            "frame unless that frame is established. If the question's frame or "
            "object binding is unresolved, copy the independent answer unchanged."
        )
    elif relation in {"front", "behind"}:
        geometry = (
            "Smaller camera depth means nearer/in front and larger camera depth "
            "means farther/behind only for a camera-relative comparison of correctly "
            "identified objects. Object-facing front/behind needs orientation that "
            "these measurements do not supply. Image y is not camera depth. If "
            "the requested frame cannot be established, copy the independent answer unchanged."
        )
    else:
        geometry = (
            "Image bbox width, height and area measure apparent extent, not physical "
            "width, height or volume. Do not invent a physical reconstruction or "
            "let an unrelated numeric field overrule the visual answer. If available "
            "measurements cannot resolve the requested property, copy the independent answer unchanged."
        )
    preserve = (
        "If the selected object, boolean conclusion or clock direction does not "
        "need changing, COPY THE INDEPENDENT ANSWER EXACTLY, including its existing "
        "object description and sentence form. Do not shorten, paraphrase, add an "
        "explanation, or normalize it just to fit a new style. The correction "
        "contract applies only when a genuine answer change is justified."
    )
    return (
        f"Original question:\n{question_text}\n\nAnswer type: {answer_type}\n"
        "Independent visual answer (not ground truth, generated without this tool evidence):\n"
        f"{json.dumps({'answer': draft}, ensure_ascii=False)}\n\n"
        f"External object evidence:\n<tool_response>{json.dumps(evidence, ensure_ascii=False)}</tool_response>\n\n"
        "Verify internally, then answer the ORIGINAL question:\n"
        "1. Identify the exact question objects, their order and the requested frame. "
        "Tool subject is only a measurement label, not necessarily the answer or "
        "first object in the question. status=ok and valid_depth_ratio do not "
        "certify correct masks. Do not invent a missing reference object.\n"
        f"2. Requested relation: {relation}. {geometry}\n"
        "3. Compare applicable evidence with what the draft actually says. Agreement "
        "is not a reason to revise it. Correct a conclusion only when correctly "
        "bound, relevant evidence and the image establish a specific contradiction. "
        "A field's presence, tool success or an unrelated numeric difference is "
        "not such a contradiction. If evidence is inapplicable, ambiguous or "
        "inconsistent, retain the independent answer rather than guessing a change.\n"
        "4. Recheck the chosen object or assertion with the same subject/reference "
        "order. A choice must select one object, not state that the tool subject "
        "does not satisfy a relation. Do not introduce a new explanation.\n\n"
        f"PRESERVATION RULE:\n{preserve}\n\n"
        f"CONTRACT FOR NECESSARY CORRECTIONS ONLY:\n{contract}\n\n"
        "Output only the final answer as ordinary text. No JSON decision, reasoning "
        "trace, tool JSON, coordinates, alternatives, or discussion of the draft."
    )

def build_mcq_no_tool_prompt(
    question_text: str,
    task_type: Optional[str],
    relation_style: bool = False,
) -> str:
    if not relation_style:
        return build_mcq_reasoning_prompt(question_text, task_type)
    q = _strip_letter_only_tail(question_text or "")
    return (
        f"Question:\n{q}\n\n"
        "Solve this step by step:\n"
        "Step 1. Identify the subject and reference objects in the question.\n"
        "Step 2. Inspect the image for their relative location/depth.\n"
        "Step 3. Decide which option best matches the visual spatial relation.\n"
        "Step 4. Pick the final option.\n\n"
        "### RESPONSE FORMAT (MANDATORY):\n"
        "You MUST write out ALL FOUR steps above, then finish with \\boxed{X} "
        "on a new line, where X is exactly one of A, B, C or D. A one-letter "
        "answer with no reasoning will be rejected."
    )

def build_mcq_system_prompt() -> str:
    return build_mcq_reasoning_system_prompt()
