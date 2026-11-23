from __future__ import annotations
import json, math, re
from .qwen3vl_runtime import generate_text, make_messages
from . import o3dvqa_protocol as protocol
from .cache import lateral_binding
from .o3dvqa_protocol import is_metric_question_name, relation_answer_contract

def clean_visible_response(text: str) -> str:
    """Remove leaked Qwen3.5 thinking/chat-template scaffolding.

    If a closed thinking block is present, the final user-visible answer is
    after the last ``</think>``. If only an unclosed ``<think>`` leaked after a
    visible answer, keep the text before the first leaked marker.
    """
    if not isinstance(text, str) or not text:
        return text
    lower = text.lower()
    close_tag = "</think>"
    open_tag = "<think>"
    if close_tag in lower:
        start = lower.rfind(close_tag) + len(close_tag)
        cleaned = text[start:].strip()
        return cleaned or text.strip()

    cut_positions = []
    for marker in ("assistant\n<think>", "assistant\r\n<think>", open_tag):
        pos = lower.find(marker)
        if pos >= 0:
            cut_positions.append(pos)
    if cut_positions:
        cleaned = text[:min(cut_positions)].strip()
        return cleaned or text.strip()
    return text.strip()

def build_vertical_prompt(question, draft, evidence):
    if evidence.get("status") != "ok" or evidence.get("evidence_type") != "estimated_ground_vertical_3d":
        raise ValueError("Only valid estimated vertical3D evidence may enter fusion")
    compact = {"coordinate_frame": evidence["coordinate_frame"], "ground_estimation": {
        key: evidence["ground_estimation"][key] for key in ("normal_camera", "inlier_fraction", "normal_disagreement_deg", "limitation", "unit")},
        "objects": {role: {key: obj[key] for key in ("prompt", "z_mean", "z_median", "z_p10", "z_p90", "valid_depth_ratio", "definition")}
                    for role, obj in evidence["objects"].items()}, "comparison": evidence["comparison"]}
    return ("Answer the original vertical-position question using the RGB image, the independent DRoPE answer, and the estimated 3D evidence.\n"
            "Z increases along an estimated background-plane normal, not camera y, image y, or camera depth. The estimate uses predicted depth, "
            "pixel intrinsics and cached SAM3 masks, with no GT depth or pose. A dominant lower-image plane may be a slope or roof; its ground identity is not certified.\n"
            "Verify the named objects and subject/reference direction against the image. Compare their visible point-cloud centroid Z values, "
            "not their top edges, screen height, physical size, or camera distance. Z percentiles describe visible surfaces, not confidence intervals.\n"
            "If binding or the estimated frame is unreliable, or there is no clear applicable contradiction, COPY THE INDEPENDENT ANSWER EXACTLY. "
            "If the selected object or yes/no conclusion remains unchanged, also copy it exactly. Otherwise return a concise corrected answer "
            "in the question's original format; preserve provided options and object names. No JSON, explanation, tool call, or mention of evidence.\n\n"
            f"Question:\n{question}\n\nIndependent 128-token DRoPE answer:\n{draft}\n\nEstimated vertical 3D evidence:\n"
            + json.dumps(compact, ensure_ascii=False, allow_nan=False))

def build_lateral_prompt(question, draft, evidence):
    if evidence.get("status") != "ok":
        raise ValueError("Only usable lateral3D evidence may enter fusion")
    compact = {key: evidence[key] for key in ("coordinate_frame", "axes", "unit", "comparison_mode", "objects", "comparison", "limitation")}
    return ("Answer the original left/right question using the RGB image, independent DRoPE answer and visible 3D object centers.\n"
            "Camera XYZ has X right, Y down and Z forward. left_coordinate=-camera_X; a larger left_coordinate means farther left. "
            "For object-to-object questions compare the two named centers; for object-to-observer questions compare the subject to camera origin zero. "
            "This camera lateral axis is unchanged by pitch-only canonicalization. No GT pose, ground plane, object heading or world yaw is inferred.\n"
            "These are means of visible SAM3-masked predicted 3D points, not screen centers, bbox sizes or complete-object centers. "
            "Check that the masks describe the objects in the question and that reference/subject and left/right are not reversed. "
            "Mean/median disagreement indicates surface sensitivity, not a calibrated uncertainty estimate.\n"
            "If binding or coordinate applicability is doubtful, or no clear applicable contradiction exists, COPY THE INDEPENDENT ANSWER EXACTLY. "
            "If the selected object or yes/no conclusion is unchanged, also copy it exactly. Otherwise give a concise corrected answer "
            "in the original question format, retaining options and object names. Do not output JSON, reasoning or tool calls.\n\n"
            f"Question:\n{question}\n\nIndependent 128-token DRoPE answer:\n{draft}\n\nLateral 3D centers:\n"
            + json.dumps(compact, ensure_ascii=False, allow_nan=False))

def build_clock_prompt(question, draft, evidence, frame):
    applicability = (
        "The horizontal frame estimates pitch from a background plane, keeping camera-left fixed. "
        "No roll, world yaw or object heading is inferred. A roof or sloped plane may bias this estimate. "
        if frame == "horizontal" else
        "This ablation is NOT pitch corrected. Do not silently treat camera-forward as horizontal-forward "
        "under a tilted view. If pitch/height ambiguity prevents a justified bearing, keep the draft. "
    )
    schema = dict(action="keep", binding_ok=False, evidence_applicable=False, contradiction="", corrected_answer="")
    return (
        "Verify a relative clock answer using the RGB image and newly measured predicted 3D displacement. "
        "The independent draft is not ground truth.\n" + CLOCK_DEFINITION + "\n" + applicability +
        "These are visible SAM3-masked predicted surfaces, not complete-object centers. Check A/B mask binding "
        "using the image; never reverse target/reference. Do not interpret image-up as forward or image-down "
        "as backward. Clock sector depends on the relative magnitudes of forward and left, not only quadrant. "
        "Mean/median direction disagreement, small separation relative to surface sensitivity, low valid-depth "
        "coverage or unstable plane support weaken the evidence; these are not calibrated confidence scores. "
        "Neither tool success nor a precise-looking vector proves the draft wrong. Correct only if image, binding "
        "and applicable geometry establish a specific contradiction. Otherwise keep the draft unchanged. "
        "No new tool call, GT pose or guessed calibration.\n\n"
        f"Question:\n{question}\n\nIndependent original 128-token DRoPE draft:\n" + json.dumps({"answer": draft}, ensure_ascii=False) +
        "\n\nPredicted relative 3D evidence:\n<tool_response>" + json.dumps(evidence, ensure_ascii=False, allow_nan=False) +
        "</tool_response>\n\nOutput JSON only: " + json.dumps(schema) +
        "\naction is keep or correct. Use actual booleans and no extra keys. For keep, contradiction and corrected_answer "
        "must be empty. For correct, binding_ok and evidence_applicable must be true, contradiction must identify an "
        "applicable geometric conflict, and corrected_answer must give one 1..12 o'clock answer in a concise sentence "
        "retaining A/B roles. Wording-only changes or the same clock slot must keep the original draft."
    )

CLOCK_DEFINITION = (
    "A is the reference/origin and B is the target. Use B minus A, retaining the camera heading, "
    "not the object's own heading. +forward is 12 o'clock, right is 3, backward is 6, left is 9. "
    "For horizontal forward/left displacement, theta = (round(degrees(atan2(-left, forward)), 2) + 360) mod 360. "
    "The dataset uses floor(theta/30), writing 0 as 12 and 1..11 as 1..11, not nearest-hour rounding. "
    "At 0,30,60,90,120,150,180,210,240,270,300,330 degrees the bins start at 12,1,2,3,4,5,6,7,8,9,10,11. "
)

def clock_evidence(cue, frame):
    ev = cue["clock3d_evidence"]
    if not cue["question_spec"]["applicable"]:
        return None
    key = "horizontal_displacement" if frame == "horizontal" else "camera_relative_displacement"
    if (frame == "horizontal" and ev["status"] != "ok") or ev.get(key, {}).get("status") != "ok":
        return None
    coordinate_key = "horizontal_forward_left" if frame == "horizontal" else "camera_forward_left"
    objects = {role: {name: obj[name] for name in ("prompt", "centroid_camera_xyz", "point_count", "valid_depth_ratio",
                                                coordinate_key + "_mean", coordinate_key + "_median")}
               for role, obj in ev["objects"].items()}
    compact = dict(reference="A: reference", target="B: subject", unit=ev["unit"], objects=objects,
                   displacement=ev[key], limitation=ev["limitation"], uses_gt_depth=False, uses_gt_pose=False)
    if frame == "horizontal":
        plane = ev["pitch_estimation"]
        compact.update(coordinate_frame=ev["horizontal_frame"], pitch_estimation={name: plane[name] for name in (
            "normal_camera", "inlier_fraction", "normal_disagreement_deg", "residual_median", "distance_threshold", "x_span", "y_span", "limitation")})
    else:
        compact.update(coordinate_frame="camera_forward_left_unrectified", axes="forward=camera_Z, left=-camera_X",
                       pitch_corrected=False, limitation="Camera-forward displacement mixes horizontal forward and height under pitch; it is NOT the dataset horizontal frame. " + ev["limitation"])
    return compact

DECISION_KEYS = {"action", "binding_ok", "evidence_applicable", "contradiction", "corrected_answer"}

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

UNIT_SCALE = {"m": (1, 1.0), "meter": (1, 1.0), "meters": (1, 1.0), "metre": (1, 1.0),
              "metres": (1, 1.0), "cm": (1, .01), "mm": (1, .001), "km": (1, 1000.),
              "m^3": (3, 1.0), "m3": (3, 1.0), "m³": (3, 1.0), "cubic meters": (3, 1.0),
              "cubic metres": (3, 1.0), "cm^3": (3, 1e-6), "cm3": (3, 1e-6), "cm³": (3, 1e-6)}

def metric_value(text):
    match = re.fullmatch(rf"\s*({NUMBER})\s*([a-zA-Z³^0-9 ]+)\s*[.]?\s*", text)
    if not match:
        return None
    unit = match[2].strip().lower()
    if unit not in UNIT_SCALE:
        return None
    value = float(match[1])
    if not math.isfinite(value) or value < 0:
        return None
    dimension, scale = UNIT_SCALE[unit]
    scaled = value * scale
    return (dimension, scaled) if math.isfinite(scaled) else None

def answer_slot(text, question, qname, evidence):
    if is_metric_question_name(qname):
        return metric_value(text)
    kind, _ = relation_answer_contract(question, qname)
    if kind == "boolean":
        match = re.fullmatch(r"\s*(Yes|No)(?:[.!]|,?\s+it\s+is(?:\s+not)?[.!]?)?\s*", text, re.I)
        return ("boolean", match[1].lower()) if match else None
    if kind == "clock_direction":
        matches = re.findall(r"(?<!\d)(1[0-2]|[1-9])\s*o['’]?\s*clock", text, re.I)
        return ("clock", matches[0]) if len(matches) == 1 else None
    if kind != "object_choice":
        return None
    normalize = lambda value: re.sub(r"\W+", " ", value.lower()).strip().removeprefix("the ")
    normalized = normalize(text)
    options = re.findall(r"(?:^|\n)([A-D])\s*[:.]\s*([^\n]+)", question)
    if options:
        matches = [key for key, value in options if normalize(value) and normalize(value) in normalized]
        if len(matches) == 1:
            return "option", matches[0]
        return None
    objects = evidence.get("objects") or {}
    matches = [key for key, obj in objects.items() if isinstance(obj, dict) and normalize(obj.get("prompt", ""))
               and normalize(obj["prompt"]) in normalized]
    return ("object", matches[0]) if len(matches) == 1 else None

def apply_clock_decision(raw, draft, question, qname, kind, evidence):
    visible = clean_visible_response(raw).strip()
    if visible.startswith("```json\n") and visible.endswith("```"):
        visible = visible[len("```json\n"):-3].strip()
    try:
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate decision key")
                result[key] = value
            return result
        decision = json.loads(visible, object_pairs_hook=unique_pairs)
        if not isinstance(decision, dict) or set(decision) != DECISION_KEYS:
            raise ValueError("Invalid decision schema")
        if (decision["action"] not in ("keep", "correct") or type(decision["binding_ok"]) is not bool
            or type(decision["evidence_applicable"]) is not bool
            or not isinstance(decision["contradiction"], str) or not isinstance(decision["corrected_answer"], str)):
            raise ValueError("Invalid decision types")
    except (ValueError, TypeError):
        return draft, {"action": "keep", "reason": "invalid_decision", "decision": None}
    audit = {"action": "keep", "reason": "model_keep", "decision": decision}
    if decision["action"] == "keep":
        if decision["contradiction"].strip() or decision["corrected_answer"].strip():
            audit["reason"] = "inconsistent_keep"
        return draft, audit
    corrected = decision["corrected_answer"].strip()
    if (not decision["binding_ok"] or not decision["evidence_applicable"]
        or not decision["contradiction"].strip() or not corrected):
        audit["reason"] = "unsupported_correction"
        return draft, audit
    if kind == "metric":
        value = metric_value(corrected)
        dimension = 3 if qname == "volume_data" else 1
        if value is None or value[0] != dimension:
            audit["reason"] = "invalid_numeric_answer"
            return draft, audit
    else:
        answer_type, _ = relation_answer_contract(question, qname)
        if answer_type == "object_choice" and re.match(r"^\s*(Yes|No)\b", corrected, re.I):
            audit["reason"] = "invalid_choice_answer"
            return draft, audit
        if answer_type in {"object_choice", "clock_direction", "boolean"} and answer_slot(corrected, question, qname, evidence) is None:
            audit["reason"] = "unresolved_answer_slot"
            return draft, audit
    old_slot = answer_slot(draft, question, qname, evidence)
    new_slot = answer_slot(corrected, question, qname, evidence)
    if corrected == draft or (old_slot is not None and new_slot == old_slot):
        audit["reason"] = "same_answer_preserve_verbatim"
        return draft, audit
    audit.update(action="correct", reason="supported_model_correction")
    return corrected, audit

SPATIAL_SYSTEM = "You answer visual spatial questions. Treat predicted geometry as fallible evidence, not ground truth. Return only the final answer."
CLOCK_SYSTEM = (
    "Verify an independent DRoPE answer using the original image and available tool evidence. "
    "The draft is not ground truth, and a successful tool is not proof of correct measurement. "
    "Preserve the original answer when its conclusion does not need changing. Correct it only "
    "for an applicable, correctly bound, explicit contradiction. Output one JSON decision, "
    "not prose or hidden reasoning. Never guess missing calibration or 3D information."
)


def resolve_spatial_answer(raw, draft, current):
    text = clean_visible_response(raw)
    if not text or "<tool_call>" in text:
        return draft, "invalid_answer"
    if text.strip() == draft.strip():
        return draft, "verbatim_drope"
    if text == current:
        return current, "verbatim_prior"
    return text, "new_answer"


def run_chain(bundle, image, sample, backend, base_evidence, draft):
    question = sample["query_question"]
    qname = (sample.get("qa_info") or {}).get("question_name")
    original_ok = base_evidence.get("status") == "ok"
    current, trace = draft, []

    def spatial(stage, prompt, visual, system=SPATIAL_SYSTEM):
        nonlocal current
        previous = current
        raw = generate_text(bundle, make_messages(visual, prompt, system), sample["depth_path"])
        current, reason = resolve_spatial_answer(raw, draft, previous)
        trace.append(dict(stage=stage, selected=True, executed=True, prior_response=previous,
                          raw_output=raw, response=current, reason=reason))

    def fallback(stage, reason):
        nonlocal current
        previous, current = current, draft
        trace.append(dict(stage=stage, selected=True, executed=False, prior_response=previous,
                          response=current, reason=reason))

    if original_ok:
        prompt = protocol.build_three_stage_fusion_prompt_v2_1(question, draft, "relation", base_evidence, qname)
        current = generate_text(bundle, make_messages(image, prompt, protocol.THREE_STAGE_FUSION_SYSTEM_V2), sample["depth_path"])
        trace.append(dict(stage="three_stage_v2_1", selected=True, executed=True, raw_output=current, response=current))
    else:
        fallback("three_stage_v2_1", "original_tool_failure")

    specs = backend.stage_specs(sample)
    if specs["vertical"]:
        ev = backend.stage_evidence(sample, "vertical") if original_ok else dict(status="failed", reason="original_tool_failure")
        if ev.get("status") == "ok":
            spatial("vertical", build_vertical_prompt(question, draft, ev), image)
        else:
            fallback("vertical", ev.get("reason", "unusable_vertical_evidence"))

    lateral_cue = None
    if specs["lateral"] is not None:
        lateral_cue = backend.stage_evidence(sample, "lateral", specs["lateral"]) if original_ok else dict(status="failed", reason="original_tool_failure")
        ev = lateral_cue.get("lateral3d_evidence", lateral_cue)
        if original_ok and ev.get("status") == "ok":
            spatial("lateral", build_lateral_prompt(question, draft, ev), image)
        else:
            fallback("lateral", ev.get("reason", "unusable_lateral_evidence"))

    if specs["clock"] is not None:
        cue = backend.stage_evidence(sample, "clock", specs["clock"]) if original_ok else None
        ev = clock_evidence(cue, "horizontal") if cue is not None and "clock3d_evidence" in cue else None
        if ev is None:
            fallback("clock_horizontal", "original_tool_failure" if not original_ok else "unusable_clock3d_evidence")
        else:
            previous = current
            raw = generate_text(bundle, make_messages(image, build_clock_prompt(question, draft, ev, "horizontal"), CLOCK_SYSTEM), sample["depth_path"])
            current, decision = apply_clock_decision(raw, draft, question, "direction_data", "relation", ev)
            trace.append(dict(stage="clock_horizontal", selected=True, executed=True, prior_response=previous,
                              raw_output=raw, response=current, decision=decision))

    if lateral_cue is not None:
        ev = lateral_cue.get("lateral3d_evidence", lateral_cue)
        if original_ok and ev.get("status") == "ok":
            overlay, binding = lateral_binding.render_binding_view(image, lateral_cue)
            try:
                evidence = lateral_binding.task_evidence(question, lateral_cue, binding)
                spatial("lateral_binding", lateral_binding.build_prompt(question, draft, evidence), overlay)
                trace[-1].update(binding_visualization=binding, task_evidence=evidence)
            finally:
                overlay.close()
        else:
            trace.append(dict(stage="lateral_binding", selected=True, executed=False,
                              response=current, reason="unchanged_ineligible_current"))
    return dict(response=current, independent_answer=draft, draft_source="generated_drope_prompt_128",
                branch="relation_tool_ok" if original_ok else "cot_fallback", fusion_stages=trace)
