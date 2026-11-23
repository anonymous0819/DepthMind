from __future__ import annotations

import re

import numpy as np

from tcot.cache.vertical_geometry import project

VERSION = "lateral3d_center_v1"
DEFAULTS = dict(min_depth=.5, max_depth=10000., min_object_points=30, min_valid_fraction=.50)


def question_spec(question, prompts):
    stem = question.split("\n", 1)[0]
    viewer = re.search(r"\b(?:of|than)\s+(?:you|me|the (?:viewer|camera))\b|\b(?:your|my)\s+(?:left|right)\b", stem, re.I)
    for prompt in sorted(prompts.values(), key=len, reverse=True):
        phrase = re.sub(r"^\s*(?:the|a|an)\s+", "", prompt, flags=re.I).strip()
        if phrase and phrase.lower() not in {"you", "me", "viewer", "camera"}:
            stem = re.sub(r"(?<!\w)(?:the\s+|a\s+|an\s+)?" + re.escape(phrase) + r"(?!\w)", " OBJECT ", stem, flags=re.I)
    if re.search(r"^\s*(?:what|how|if\b|where\b)|\b(?:measure|approximate|o['’]?clock)\b", stem, re.I):
        return None
    if re.search(r"\b(?:appear|appears|positioned|located|is|are)\s+(?:(?:directly|more|further)\s+)?(?:above|below|under|over|behind|in front)\b|\b(?:shorter|taller|wider|thinner|bigger|smaller)\s+than\b", stem, re.I):
        return None
    if re.search(r"\bOBJECT\s+(?:is\s+)?(?:above|below|under|over|behind|in front)\b|\b(?:above|below)\s*\?|\b(?:positioned|located)\s+(?:higher|lower)\b|\b(?:direction|angle)\s+of\b|\b(?:more|less|lesser|greater)\s+(?:height|width|volume|distant)\b|\b(?:further|nearer)\s+to\s+the\s+viewer\b", stem, re.I):
        return None
    match = re.search(r"\b(?:to|on|at)\s+(?:the\s+)?(left|right)(?:\s+(?:side|hand))?\b|\b(left|right)\s+of\b|\b(?:more|further|farther)\s+(?:to\s+(?:the\s+)?)?(left|right)\b", stem, re.I)
    if not match:
        return None
    family = next(value for value in match.groups() if value).lower()
    orientation = bool(re.search(r"\b(?:facing|heading|orientation|north|south|east|west)\b", stem, re.I))
    image_frame = bool(re.search(r"\b(?:image|screen|photo|picture|pixel)\b", stem, re.I))
    return dict(relation=family, comparison="object_to_observer" if viewer else "object_to_object",
                required_roles=["subject"] if viewer else ["subject", "reference"],
                applicable=not (orientation or image_frame),
                reason="unsupported_facing_or_image_frame" if orientation or image_frame else "camera_lateral_comparison")


def center(mask, depth, intrinsics, valid, prompt):
    selected = mask & valid
    fraction = float(selected.sum() / max(int(mask.sum()), 1))
    if selected.sum() < DEFAULTS["min_object_points"] or fraction < DEFAULTS["min_valid_fraction"]:
        return dict(status="failed", reason="insufficient_object_depth", prompt=prompt, valid_depth_ratio=fraction)
    points, _, _ = project(depth, intrinsics, selected)
    centroid = points.mean(axis=0)
    return dict(status="ok", prompt=prompt, centroid_camera_xyz=centroid.tolist(),
                left_coordinate=-float(centroid[0]), lateral_median=-float(np.median(points[:, 0])),
                point_count=len(points), valid_depth_ratio=fraction,
                definition="Mean of all visible masked predicted 3D points; not complete-object center or mask identity confidence.")


def build_evidence(depth, intrinsics, valid, masks, prompts, spec):
    result = dict(version=VERSION, evidence_type="camera_lateral_3d_centers", status="failed",
                  uses_gt_depth=False, uses_gt_pose=False, coordinate_frame="camera_origin_lateral",
                  axes="camera XYZ: X right, Y down, Z forward; left_coordinate=-camera_X",
                  unit="predicted_depth_units", comparison_mode=spec["comparison"], objects={},
                  limitation="No ground plane or camera pose is used. Left/right assumes camera-aligned horizontal axes, not object facing or world yaw.")
    if not spec["applicable"]:
        result["reason"] = spec["reason"]
        return result
    roles = spec["required_roles"]
    if any(role not in masks or not prompts.get(role) for role in roles):
        result["reason"] = "missing_required_masks_or_prompts"
        return result
    for role in roles:
        result["objects"][role] = center(masks[role], depth, intrinsics, valid, prompts[role])
    if any(obj["status"] != "ok" for obj in result["objects"].values()):
        result["reason"] = "invalid_object_3d"
        return result
    subject = result["objects"]["subject"]["left_coordinate"]
    reference = result["objects"].get("reference", {}).get("left_coordinate", 0.)
    difference = subject - reference
    result.update(status="ok", reason="lateral_centroids_available", comparison=dict(
        subject_minus_reference_left=difference, reference_origin="camera_origin" if len(roles) == 1 else "reference_object",
        ordering="subject_left" if difference > 0 else "subject_right" if difference < 0 else "equal_centroids"))
    return result
