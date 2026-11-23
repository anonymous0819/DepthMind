from __future__ import annotations

import json
import re

from .qwen3vl_runtime import generate_independent_answer, generate_text, make_messages


def answer_contract(question, question_name):
    qname = question_name or ""
    if qname in {"direction_data", "direction2agent"}:
        return "clock_direction", (
            "Return exactly one integer clock position from 1 to 12, such as \"5 o'clock\". "
            "Do not output degrees, coordinates, two directions, or a Yes/No answer."
        )
    if qname.endswith(("_choice", "_multichoice")) or re.match(r"\s*(?:which\b|who\b|what\s+object\b)", question, re.I):
        return "object_choice", (
            "This is an OBJECT-CHOICE question, NOT a yes/no question. Return the full "
            "description of exactly one selected object from the question; include its "
            "distinguishing attributes. If the question supplies Same/Unknown options, "
            "select that option only when justified. Do not return a bare option letter. "
            "Never begin with Yes or No, and never replace the choice with a negative "
            "statement about the tool subject. If subject does not satisfy the requested "
            "comparison, explicitly select the reference when the comparison supports it."
        )
    if qname.endswith(("_predicate", "_tfqa", "_relationship2agent")) or re.match(r"\s*(?:is|are|does|do|can|could|would)\b", question, re.I):
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


def infer_direction_roles(question):
    pair = re.fullmatch(r"If you are at (.+),\s*where will you find (.+?)\?\s*", question.strip(), re.I)
    if pair:
        return dict(setting="object_relative", reference=pair[1].strip(), subject=pair[2].strip())
    relative = re.search(r"\b(?:direction|angle)\s+of\s+(.+?)\s+(?:relative to|from)\s+(.+?)\s*[?.]?\s*$", question, re.I)
    if relative:
        reference = relative[2].strip()
        viewer = reference.lower().removeprefix("the ") in {"you", "me", "camera", "viewer", "agent"}
        return dict(setting="ego" if viewer else "object_relative", reference="viewer/agent" if viewer else reference, subject=relative[1].strip())
    ego = re.search(r"\b(?:direction|angle)\s+of\s+(.+?)\s*[?.]?\s*$", question, re.I)
    if ego and not re.search(r"\b(?:relative|from|compared|with respect)\b", ego[1], re.I):
        return dict(setting="ego", reference="viewer/agent", subject=ego[1].strip())
    return None


def build_relation_cues_prompt(question, draft, evidence, question_name=None):
    """The single relation-cues entry point, conditioned on usable geometry."""
    if evidence.get("status") == "failed":
        raise ValueError("Failed evidence must not enter fusion")
    if "relative_measurements" in evidence and "binding_visualization" in evidence:
        return (
            "Answer the original left/right question by integrating the same RGB scene, the independent DRoPE answer "
            "and measured geometry. This is the tool-evidence fusion stage after the unchanged initial DRoPE inference.\n"
            "The image has a light mask overlay: A/orange is the cue subject, B/cyan the cue reference if present. "
            "It retains the original image size and camera layout. Mask outlines, bbox and 2D centers are binding aids ONLY; "
            "do not use their image position as a substitute for the predicted spatial relation.\n"
            "Reason through these checks before answering:\n"
            "1. Bind each labeled visible mask to the named object in the question using its appearance and extent. "
            "A/B denote measured identities, not necessarily the grammatical first/second object. Multiple instances, "
            "overlapping masks or incomplete surfaces can invalidate this binding.\n"
            "2. Resolve the actual predicate and reference: which object is being asked about, compared to which other "
            "object or the viewer. Camera-lateral geometry is not object-facing/world-yaw geometry; explicit image-frame "
            "questions require image evidence, not pretending camera X is image x.\n"
            "3. Use the measured camera X centers and A-minus-B delta to infer the applicable relation. "
            "Positive X is right. If the query reverses A/B, reverse the comparison, not the labels. Check the median "
            "and visible-surface limitations. Do not infer left/right from depth mean, bbox width or color labels.\n"
            "4. Compare this supported relation with the initial answer. The initial answer is NOT privileged ground truth: "
            "when binding and reference frame are appropriate and the evidence gives a clear contradiction, reconsider it "
            "instead of repeating it by default. Conversely numerical precision alone does not prove applicability. "
            "If binding/frame/evidence is doubtful, keep the original answer.\n"
            "5. Output only the final answer in the original format, including named objects/options in a concise relation "
            "sentence. No reasoning, JSON or tool calls. If the conclusion/selected object is unchanged, copy the initial "
            "answer verbatim; do not replace an object sentence with bare Yes/No.\n\n"
            f"Original question:\n{question}\n\nIndependent original 128-token DRoPE answer:\n{draft}\n\n"
            "Binding and task-aligned measured evidence:\n" + json.dumps(evidence, ensure_ascii=False, allow_nan=False)
        )
    if evidence.get("evidence_type") == "estimated_ground_vertical_3d":
        compact = {"coordinate_frame": evidence["coordinate_frame"], "ground_estimation": {
            key: evidence["ground_estimation"][key] for key in ("normal_camera", "inlier_fraction", "normal_disagreement_deg", "limitation", "unit")},
            "objects": {role: {key: obj[key] for key in ("prompt", "z_mean", "z_median", "z_p10", "z_p90", "valid_depth_ratio", "definition")}
                        for role, obj in evidence["objects"].items()}, "comparison": evidence["comparison"]}
        return (
            "Answer the original vertical-position question using the RGB image, the independent DRoPE answer, and the estimated 3D evidence.\n"
            "Z increases along an estimated background-plane normal, not camera y, image y, or camera depth. The estimate uses predicted depth, "
            "pixel intrinsics and cached SAM3 masks, with no GT depth or pose. A dominant lower-image plane may be a slope or roof; its ground identity is not certified.\n"
            "Verify the named objects and subject/reference direction against the image. Compare their visible point-cloud centroid Z values, "
            "not their top edges, screen height, physical size, or camera distance. Z percentiles describe visible surfaces, not confidence intervals.\n"
            "If binding or the estimated frame is unreliable, or there is no clear applicable contradiction, COPY THE INDEPENDENT ANSWER EXACTLY. "
            "If the selected object or yes/no conclusion remains unchanged, also copy it exactly. Otherwise return a concise corrected answer "
            "in the question's original format; preserve provided options and object names. No JSON, explanation, tool call, or mention of evidence.\n\n"
            f"Question:\n{question}\n\nIndependent 128-token DRoPE answer:\n{draft}\n\nEstimated vertical 3D evidence:\n"
            + json.dumps(compact, ensure_ascii=False, allow_nan=False)
        )
    if evidence.get("evidence_type") == "predicted_pitch_horizontal_relative_3d":
        compact = {key: value for key, value in evidence.items() if key not in {"status", "evidence_type"}}
        definition = (
            "A is the reference/origin and B is the target. Use B minus A, retaining the camera heading, "
            "not the object's own heading. +forward is 12 o'clock, right is 3, backward is 6, left is 9. "
            "For horizontal forward/left displacement, theta = (round(degrees(atan2(-left, forward)), 2) + 360) mod 360. "
            "The dataset uses floor(theta/30), writing 0 as 12 and 1..11 as 1..11, not nearest-hour rounding. "
            "At 0,30,60,90,120,150,180,210,240,270,300,330 degrees the bins start at 12,1,2,3,4,5,6,7,8,9,10,11. "
        )
        schema = dict(action="keep", binding_ok=False, evidence_applicable=False, contradiction="", corrected_answer="")
        return (
            "Verify a relative clock answer using the RGB image and newly measured predicted 3D displacement. "
            "The independent draft is not ground truth.\n" + definition + "\n"
            "The horizontal frame estimates pitch from a background plane, keeping camera-left fixed. "
            "No roll, world yaw or object heading is inferred. A roof or sloped plane may bias this estimate. "
            "These are visible SAM3-masked predicted surfaces, not complete-object centers. Check A/B mask binding "
            "using the image; never reverse target/reference. Do not interpret image-up as forward or image-down "
            "as backward. Clock sector depends on the relative magnitudes of forward and left, not only quadrant. "
            "Mean/median direction disagreement, small separation relative to surface sensitivity, low valid-depth "
            "coverage or unstable plane support weaken the evidence; these are not calibrated confidence scores. "
            "Neither tool success nor a precise-looking vector proves the draft wrong. Correct only if image, binding "
            "and applicable geometry establish a specific contradiction. Otherwise keep the draft unchanged. "
            "No new tool call, GT pose or guessed calibration.\n\n"
            f"Question:\n{question}\n\nIndependent original 128-token DRoPE draft:\n" + json.dumps({"answer": draft}, ensure_ascii=False)
            + "\n\nPredicted relative 3D evidence:\n<tool_response>" + json.dumps(compact, ensure_ascii=False, allow_nan=False)
            + "</tool_response>\n\nOutput JSON only: " + json.dumps(schema)
            + "\naction is keep or correct. Use actual booleans and no extra keys. For keep, contradiction and corrected_answer "
            "must be empty. For correct, binding_ok and evidence_applicable must be true, contradiction must identify an "
            "applicable geometric conflict, and corrected_answer must give one 1..12 o'clock answer in a concise sentence "
            "retaining A/B roles. Wording-only changes or the same clock slot must keep the original draft."
        )
    if evidence.get("status") != "ok" or evidence.get("evidence_type") != "object_geometry":
        raise ValueError("Unsupported relation evidence schema")
    answer_type, contract = answer_contract(question, question_name)
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
        match = re.search(r"\b(left|right|above|below|front|behind)\b", question, re.I)
        relation = match[1].lower() if match else "question-defined"
    if answer_type == "clock_direction":
        roles = infer_direction_roles(question)
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
        f"Original question:\n{question}\n\nAnswer type: {answer_type}\n"
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


def clock_slot(answer):
    slots = re.findall(r"\b(1[0-2]|[1-9])\s*o['’]?clock\b", answer, re.I)
    if not slots and re.fullmatch(r"\s*(?:1[0-2]|[1-9])\s*", answer):
        return int(answer.strip())
    return int(slots[0]) if len(set(slots)) == 1 else None


def resolve_clock_decision(raw, draft):
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate decision key")
            obj[key] = value
        return obj

    visible = raw.strip()
    if visible.startswith("```json\n") and visible.endswith("```"):
        visible = visible[len("```json\n"):-3].strip()
    try:
        decision = json.loads(visible, object_pairs_hook=unique_pairs)
        if not isinstance(decision, dict) or set(decision) != {"action", "binding_ok", "evidence_applicable", "contradiction", "corrected_answer"}:
            raise ValueError("Invalid decision schema")
        if (decision["action"] not in {"keep", "correct"} or type(decision["binding_ok"]) is not bool
            or type(decision["evidence_applicable"]) is not bool or not isinstance(decision["contradiction"], str)
            or not isinstance(decision["corrected_answer"], str)):
            raise ValueError("Invalid decision types")
    except (ValueError, TypeError):
        return draft, "invalid_decision"
    if decision["action"] == "keep":
        return draft, "model_keep"
    corrected = decision["corrected_answer"].strip()
    if not decision["binding_ok"] or not decision["evidence_applicable"] or not decision["contradiction"].strip() or clock_slot(corrected) is None:
        return draft, "unsupported_correction"
    if corrected == draft or clock_slot(corrected) == clock_slot(draft):
        return draft, "same_answer_preserve_verbatim"
    return corrected, "supported_model_correction"


def run(bundle, image, question, question_name, depth_path, evidence=None, fusion_image=None):
    draft = generate_independent_answer(bundle, image, question, depth_path)
    usable = evidence is not None and evidence.get("status") != "failed" and bool(evidence.get("objects") or evidence.get("visible_surface_measurements"))
    if not usable:
        return dict(response=draft, direct_response=draft, branch="cot_fallback", fusion_executed=False)
    prompt = build_relation_cues_prompt(question, draft, evidence, question_name)
    system = (
        "Answer the original spatial question using its image, an independent draft, "
        "and external object evidence. Verify object identity, reference frame, "
        "comparison direction and answer type before changing the draft. Tool success "
        "does not guarantee correct object binding or a reason to change an answer. "
        "A choice question requires one selected object, never a Yes/No assertion. "
        "Follow the explicit final-output contract and output only the final answer."
    )
    if "binding_visualization" in evidence or evidence.get("evidence_type") == "estimated_ground_vertical_3d":
        system = "You answer visual spatial questions. Treat predicted geometry as fallible evidence, not ground truth. Return only the final answer."
    elif evidence.get("evidence_type") == "predicted_pitch_horizontal_relative_3d":
        system = (
            "Verify an independent DRoPE answer using the original image and available tool evidence. "
            "The draft is not ground truth, and a successful tool is not proof of correct measurement. "
            "Preserve the original answer when its conclusion does not need changing. Correct it only "
            "for an applicable, correctly bound, explicit contradiction. Output one JSON decision, "
            "not prose or hidden reasoning. Never guess missing calibration or 3D information."
        )
    messages = make_messages(image if fusion_image is None else fusion_image, prompt, system)
    raw = generate_text(bundle, messages, depth_path)
    response, reason = raw.strip(), "evidence_guided_answer"
    if evidence.get("evidence_type") == "predicted_pitch_horizontal_relative_3d":
        response, reason = resolve_clock_decision(raw, draft)
    elif (not response or response.startswith(("{", "["))
          or re.search(r"<\s*/?\s*tool_call\b|```", response, re.I)):
        response, reason = draft, "invalid_answer"
    return dict(response=response, response_raw=raw, direct_response=draft, branch="relation_tool_ok",
                fusion_executed=True, fusion_reason=reason, relation_evidence=evidence)
