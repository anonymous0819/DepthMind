from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tcot.cache.vertical_geometry import load_mask

VERSION = "lateral_binding_replay_v2"
COLORS = {"subject": (255, 160, 32), "reference": (32, 200, 255)}
LABELS = {"subject": "A", "reference": "B"}
OVERLAY_ALPHA = .12


def role_text(question, objects):
    text = question.split("\n", 1)[0]
    phrases = [(role, re.sub(r"^\s*(?:the|a|an)\s+", "", obj["prompt"], flags=re.I).strip())
               for role, obj in objects.items()]
    if len({phrase.casefold() for _, phrase in phrases}) != len(phrases):
        return None
    for role, phrase in sorted(phrases, key=lambda pair: len(pair[1]), reverse=True):
        text = re.sub(r"(?<!\w)(?:(?:the|a|an)\s+)?" + re.escape(phrase) + r"(?!\w)",
                      role.upper(), text, flags=re.I)
    return text


def render_binding_view(image, cue):
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb).copy()
    masks = {}
    objects = {}
    for role, obj in cue["lateral3d_evidence"]["objects"].items():
        mask = load_mask(cue["source"]["masks"][role]["path"], (rgb.height, rgb.width))
        if not mask.any():
            raise ValueError("Certified usable object mask is empty in RGB frame")
        masks[role] = mask
        y, x = np.nonzero(mask)
        bbox = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
        objects[role] = dict(label=LABELS[role], prompt=obj["prompt"], color_rgb=list(COLORS[role]),
            mask_area_pixels=int(mask.sum()), bbox_xyxy_normalized=[bbox[0]/rgb.width, bbox[1]/rgb.height, bbox[2]/rgb.width, bbox[3]/rgb.height],
            center_xy_normalized=[float(x.mean()/rgb.width), float(y.mean()/rgb.height)], bbox_pixels=bbox)
    occupied = np.zeros((rgb.height, rgb.width), bool)
    color_sum = np.zeros_like(pixels, dtype=float)
    counts = np.zeros((rgb.height, rgb.width), int)
    for role, mask in masks.items():
        occupied |= mask
        color_sum[mask] += COLORS[role]
        counts[mask] += 1
    tint = color_sum[occupied] / counts[occupied, None]
    pixels[occupied] = np.rint((1-OVERLAY_ALPHA)*pixels[occupied] + OVERLAY_ALPHA*tint).astype(np.uint8)
    overlay = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for role, mask in masks.items():
        padded = np.pad(mask, 1)
        interior = (padded[1:-1,1:-1] & padded[:-2,1:-1] & padded[2:,1:-1]
                    & padded[1:-1,:-2] & padded[1:-1,2:])
        boundary = mask & ~interior
        boundary_y, boundary_x = np.nonzero(boundary)
        draw.point(list(zip(boundary_x.tolist(), boundary_y.tolist())), fill=COLORS[role])
        obj = objects[role]
        x0, y0, x1, y1 = obj.pop("bbox_pixels")
        draw.rectangle((x0,y0,x1-1,y1-1), outline=COLORS[role], width=1)
        label = obj["label"]
        label_box = draw.textbbox((0,0), label, font=font)
        width, height = label_box[2]-label_box[0]+8, label_box[3]-label_box[1]+6
        lx, ly = min(x0, max(0,rgb.width-width)), max(0,y0-height)
        draw.rectangle((lx,ly,lx+width-1,ly+height-1), fill=(16,16,16))
        draw.text((lx+4,ly+3-label_box[1]), label, font=font, fill=COLORS[role])
    overlap = int((counts > 1).sum())
    info = dict(version=VERSION, objects=objects, rgb_size=[rgb.width,rgb.height], mask_resize="nearest_to_same_RGB_frame",
        overlap_pixels=overlap, overlay_alpha=OVERLAY_ALPHA, overlay_pixel_sha256=hashlib.sha256(overlay.tobytes()).hexdigest(),
        input_policy="Single same-size RGB with light SAM3 tint, outlines and labels; no crop, panel, rescale or second image. Original depth-grid alignment retained.",
        limitation="Colored masks and boxes are for identity verification only, not geometry, facing, depth confidence or GT.")
    return overlay, info


def task_evidence(question, cue, binding):
    ev = cue["lateral3d_evidence"]
    if ev["status"] != "ok":
        raise ValueError("Only usable lateral evidence enters binding fusion")
    objects = ev["objects"]
    transformed = role_text(question, objects)
    mentions = re.findall(r"\b(?:SUBJECT|REFERENCE)\b", transformed) if transformed is not None else []
    question_order = [{"label": LABELS[role.lower()], "cue_role": role.lower()} for role in mentions]
    reference = objects.get("reference")
    mean_right = -float(ev["comparison"]["subject_minus_reference_left"])
    median_right = -(float(objects["subject"]["lateral_median"]) - float(reference["lateral_median"] if reference else 0.))
    measurements = {}
    for role, obj in objects.items():
        measurements[LABELS[role]] = dict(cue_role=role, prompt=obj["prompt"], centroid_camera_xyz=obj["centroid_camera_xyz"],
            camera_x_median=-obj["lateral_median"], point_count=obj["point_count"], valid_depth_ratio=obj["valid_depth_ratio"],
            visible_center_mean_median_x_difference=abs(obj["centroid_camera_xyz"][0]+obj["lateral_median"]))
    return dict(binding_visualization=binding, question_alignment=dict(original_question=question,
        relation=cue["question_spec"]["relation"], comparison_mode=ev["comparison_mode"],
        cue_object_labels="A=subject; B=reference when present. Labels are identities, NOT answers or guaranteed grammatical roles.",
        question_with_cue_roles=transformed, role_mentions_in_question_order=question_order,
        limitation="Use the original full question to identify who is compared to whom; role mentions alone do not parse predicate argument order."),
        coordinate_frame="camera lateral; X positive right, Y down, Z camera-forward", unit=ev["unit"],
        visible_surface_measurements=measurements,
        relative_measurements=dict(displacement_definition="A minus B" if reference else "A minus camera origin zero",
            delta_camera_x_mean=mean_right, delta_camera_x_median=median_right,
            normalized_axis_difference=(mean_right/max(abs(mean_right),abs(median_right))) if max(abs(mean_right),abs(median_right))>0 else 0.,
            mean_median_same_sign=mean_right*median_right>0,
            sign_semantics="Positive delta_camera_x places A's predicted visible center to the right of B/origin; negative to the left. Reverse the subtraction if the question asks about B relative to A."),
        uses_gt_depth=False, uses_gt_pose=False,
        limitation=ev["limitation"] + " Visible masked centers are not complete-object centers. Agreement of mean/median is not calibrated confidence. No program-supplied Yes/No or selected object.")




def build_prompt(question, draft, evidence):
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
