from __future__ import annotations

import re
import numpy as np
from tcot.cache import vertical_geometry as vertical

VERSION = "clock_horizontal3d_v1"
DEFAULTS = dict(min_depth=.5, max_depth=10000., min_object_points=30, min_valid_fraction=.50,
                min_forward_axis_norm=.15, plane=vertical.DEFAULTS.copy())


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().casefold()).removeprefix("the ")


def question_spec(question, prompts):
    match = re.fullmatch(r"\s*If you are at (.+?), where will you find (.+?)\?\s*", question, re.I)
    if match is None:
        return dict(applicable=False, reason="unsupported_clock_template", required_roles=["reference", "subject"])
    reference, target = match.groups()
    bound = (normalize(reference) == normalize(prompts.get("reference", ""))
             and normalize(target) == normalize(prompts.get("subject", "")))
    return dict(applicable=bound, reason="bound_reference_A_target_B" if bound else "question_object_binding_mismatch",
                reference_description=reference, target_description=target, required_roles=["reference", "subject"])


def object_points(mask, depth, intrinsics, valid, prompt):
    selected = mask & valid
    count = int(selected.sum())
    ratio = float(count / max(int(mask.sum()), 1))
    metadata = dict(prompt=prompt, point_count=count, valid_depth_ratio=ratio)
    if count < DEFAULTS["min_object_points"] or ratio < DEFAULTS["min_valid_fraction"]:
        return None, dict(**metadata, status="failed", reason="insufficient_object_depth")
    points, _, _ = vertical.project(depth, intrinsics, selected)
    return points, dict(**metadata, status="ok", centroid_camera_xyz=points.mean(axis=0).tolist(),
                        definition="Mean of visible masked predicted 3D points, not complete-object center.")


def displacement(objects, frame):
    reference, target = objects["reference"], objects["subject"]
    mean = np.asarray(target[frame + "_mean"]) - np.asarray(reference[frame + "_mean"])
    median = np.asarray(target[frame + "_median"]) - np.asarray(reference[frame + "_median"])
    magnitude = float(np.linalg.norm(mean))
    median_magnitude = float(np.linalg.norm(median))
    if magnitude <= 1e-12 or median_magnitude <= 1e-12:
        return dict(status="failed", reason="coincident_horizontal_centers")
    cosine = float(np.clip(mean @ median / (magnitude * median_magnitude), -1., 1.))
    sensitivity = sum(float(np.linalg.norm(np.asarray(obj[frame + "_mean"]) - np.asarray(obj[frame + "_median"])))
                      for obj in objects.values())
    return dict(status="ok", target_minus_reference_forward=float(mean[0]), target_minus_reference_left=float(mean[1]),
                median_target_minus_reference_forward=float(median[0]), median_target_minus_reference_left=float(median[1]),
                normalized_forward_left=(mean / magnitude).tolist(), median_normalized_forward_left=(median / median_magnitude).tolist(),
                horizontal_separation=magnitude, mean_median_direction_cosine=cosine,
                surface_sensitivity=sensitivity, separation_exceeds_surface_sensitivity=magnitude > sensitivity,
                limitation="Mean/median agreement and surface sensitivity are diagnostics, not calibrated confidence.")


def build_evidence(depth, intrinsics, valid, masks, prompts, spec):
    output = dict(version=VERSION, evidence_type="predicted_pitch_horizontal_relative_3d", status="failed",
                  uses_gt_depth=False, uses_gt_pose=False, unit="predicted_depth_units",
                  reference="A: reference object", target="B: subject object", displacement="target B minus reference A",
                  camera_axes="X right, Y down, Z camera-forward", objects={},
                  limitation="Predicted depth/K and visible SAM3 surfaces. Background plane estimates pitch, not certified gravity; roofs/slopes may bias it. No world yaw or object facing is inferred.")
    if not spec["applicable"]:
        output["reason"] = spec["reason"]
        return output
    if any(role not in masks or not prompts.get(role) for role in ("reference", "subject")):
        output["reason"] = "missing_required_masks_or_prompts"
        return output
    points_by_role = {}
    for role in ("reference", "subject"):
        points, meta = object_points(masks[role], depth, intrinsics, valid, prompts[role])
        output["objects"][role] = meta
        if points is not None:
            points_by_role[role] = points
    if len(points_by_role) != 2:
        output["reason"] = "invalid_object_3d"
        return output
    for role, points in points_by_role.items():
        coords = np.column_stack((points[:, 2], -points[:, 0]))
        output["objects"][role].update(camera_forward_left_mean=coords.mean(axis=0).tolist(),
                                         camera_forward_left_median=np.median(coords, axis=0).tolist())
    output["camera_relative_displacement"] = displacement(output["objects"], "camera_forward_left")
    plane = vertical.estimate_ground(depth, intrinsics, valid, masks["reference"] | masks["subject"], DEFAULTS["plane"])
    output["pitch_estimation"] = plane
    if plane["status"] != "ok":
        output["reason"] = plane["reason"]
        return output
    normal = np.asarray(plane["normal_camera"], dtype=float)
    forward = np.array([0., normal[2], -normal[1]])
    length = float(np.linalg.norm(forward))
    if length < DEFAULTS["min_forward_axis_norm"]:
        output["reason"] = "unresolved_camera_heading_on_horizontal_plane"
        return output
    forward /= length
    left = np.array([-1., 0., 0.])
    up = np.cross(forward, left)
    output["horizontal_frame"] = dict(coordinate_frame="predicted_pitch_only_camera_heading",
        forward_axis_camera=forward.tolist(), left_axis_camera=left.tolist(), up_axis_camera=up.tolist(),
        axes="+forward, +left; translate origin to A, retain camera heading",
        pitch_only=True, roll_corrected=False, world_yaw_used=False,
        normal_lateral_component=float(normal[0]),
        definition="Project estimated up normal to camera YZ to estimate pitch only; keep camera left axis unchanged. This is not full plane canonicalization or GT pose.")
    basis = np.stack((forward, left))
    for role, points in points_by_role.items():
        coords = points @ basis.T
        output["objects"][role].update(horizontal_forward_left_mean=coords.mean(axis=0).tolist(),
                                         horizontal_forward_left_median=np.median(coords, axis=0).tolist())
    comparison = displacement(output["objects"], "horizontal_forward_left")
    output["horizontal_displacement"] = comparison
    if comparison["status"] != "ok":
        output["reason"] = comparison["reason"]
        return output
    output.update(status="ok", reason="predicted_horizontal_relative_vector_available")
    return output
