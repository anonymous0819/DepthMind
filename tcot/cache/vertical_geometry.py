from __future__ import annotations

import numpy as np
from PIL import Image

from tcot.cache.relation_evidence_guard import question_context

VERSION = "vertical3d_ground_v1"
DEFAULTS = dict(seed=1729, min_depth=.5, max_depth=10000., lower_fraction=.55,
                max_points=5000, iterations=256, distance_fraction=.01,
                min_points=200, min_inlier_fraction=.30, min_x_span=.30,
                min_y_span=.12, min_up_component=.20, max_forward_component=.10,
                max_normal_disagreement_deg=10., min_mask_valid_fraction=.50)


def vertical_question(question, prompts):
    evidence = {"objects": {role: {"prompt": prompt} for role, prompt in prompts.items()}}
    context = question_context(question, None, evidence)
    if context["family"] not in {"above", "below"}:
        return False
    return not context["explicit_image_frame"]


def resize_mask(mask, shape):
    if mask.shape == shape:
        return mask.astype(bool)
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((shape[1], shape[0]), Image.Resampling.NEAREST)).astype(bool)


def load_mask(path, shape):
    with np.load(path, allow_pickle=False) as data:
        if "masks" not in data.files and "mask" not in data.files:
            raise ValueError("Missing named SAM3 mask field")
        mask = data["masks" if "masks" in data.files else "mask"].astype(bool)
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    if mask.ndim != 2:
        raise ValueError("Invalid SAM3 mask dimensions")
    return resize_mask(mask, shape)


def project(depth, intrinsics, valid):
    k = np.asarray(intrinsics, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-6 or abs(k[1, 0]) > 1e-6):
        raise ValueError("Require pixel intrinsics for this exact depth resolution")
    y, x = np.nonzero(valid)
    z = depth[y, x].astype(np.float64)
    points = np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))
    return points, y, x


def fit_plane(points):
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[-1]
    if normal[1] > 0:
        normal = -normal
    return normal, -float(normal @ center)


def estimate_ground(depth, intrinsics, valid, excluded, config=None):
    cfg = DEFAULTS if config is None else config
    region = valid & ~excluded
    region[:int(depth.shape[0] * cfg["lower_fraction"])] = False
    points, y, x = project(depth, intrinsics, region)
    if len(points) < cfg["min_points"]:
        return dict(status="failed", reason="insufficient_background_points", candidate_points=len(points))
    rng = np.random.default_rng(cfg["seed"])
    chosen = rng.choice(len(points), min(len(points), cfg["max_points"]), replace=False)
    points, y, x = points[chosen], y[chosen], x[chosen]
    scale = float(np.median(points[:, 2]))
    threshold = cfg["distance_fraction"] * scale
    triples = rng.integers(0, len(points), size=(cfg["iterations"], 3))
    sampled = points[triples]
    normals = np.cross(sampled[:, 1] - sampled[:, 0], sampled[:, 2] - sampled[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    normals[normals[:, 1] > 0] *= -1
    offsets = -np.einsum("ij,ij->i", normals, sampled[:, 0])
    eligible = ((lengths > 1e-8) & (normals[:, 1] <= -cfg["min_up_component"])
                & (normals[:, 2] <= cfg["max_forward_component"]) & (offsets > 0))
    normals, offsets = normals[eligible], offsets[eligible]
    if not len(normals):
        return dict(status="failed", reason="no_ground_candidate_under_camera_prior")
    residual = np.abs(np.einsum("ij,kj->ik", points, normals) + offsets)
    counts = (residual < threshold).sum(axis=0)
    best = int(counts.argmax())
    fraction = float(counts[best] / len(points))
    if fraction < cfg["min_inlier_fraction"]:
        return dict(status="failed", reason="weak_ground_support", inlier_fraction=fraction)
    close = counts >= counts[best] * .9
    angles = np.degrees(np.arccos(np.clip(normals @ normals[best], -1, 1)))
    if np.any(close & (angles > cfg["max_normal_disagreement_deg"])):
        return dict(status="failed", reason="ambiguous_ground_orientation", inlier_fraction=fraction)
    if np.any(close & (np.abs(offsets - offsets[best]) > 3 * threshold) & (angles < 5)):
        return dict(status="failed", reason="ambiguous_parallel_planes", inlier_fraction=fraction)
    inliers = residual[:, best] < threshold
    for _ in range(2):
        if inliers.sum() < cfg["min_points"]:
            return dict(status="failed", reason="insufficient_plane_inliers")
        normal, offset = fit_plane(points[inliers])
        inliers = np.abs(points @ normal + offset) < threshold
    if normal[1] > -cfg["min_up_component"] or normal[2] > cfg["max_forward_component"] or offset <= 0:
        return dict(status="failed", reason="refined_plane_violates_camera_prior")
    iy, ix = y[inliers], x[inliers]
    fraction = float(inliers.mean())
    xspan = float((np.percentile(ix, 95) - np.percentile(ix, 5)) / depth.shape[1])
    yspan = float((np.percentile(iy, 95) - np.percentile(iy, 5)) / depth.shape[0])
    if fraction < cfg["min_inlier_fraction"] or xspan < cfg["min_x_span"] or yspan < cfg["min_y_span"]:
        return dict(status="failed", reason="insufficient_plane_spatial_support", inlier_fraction=fraction, x_span=xspan, y_span=yspan)
    left, right = np.where(inliers)[0][::2], np.where(inliers)[0][1::2]
    n1, _ = fit_plane(points[left])
    n2, _ = fit_plane(points[right])
    disagreement = float(np.degrees(np.arccos(np.clip(n1 @ n2, -1, 1))))
    if disagreement > cfg["max_normal_disagreement_deg"]:
        return dict(status="failed", reason="unstable_ground_normal", normal_disagreement_deg=disagreement)
    axis_x = np.array([1., 0., 0.]) - normal * normal[0]
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    origin = -offset * normal
    return dict(status="ok", reason="estimated_background_plane_under_camera_prior", normal_camera=normal.tolist(),
                plane_offset=offset, origin_camera=origin.tolist(), axes_camera=[axis_x.tolist(), axis_y.tolist(), normal.tolist()],
                inlier_fraction=fraction, residual_median=float(np.median(np.abs(points[inliers] @ normal + offset))),
                distance_threshold=threshold, depth_scale=scale, x_span=xspan, y_span=yspan,
                normal_disagreement_deg=disagreement, candidate_points=len(points), inlier_points=int(inliers.sum()),
                limitation="Lower-image dominant plane and camera-up prior; not a certified ground/gravity estimate. Roofs/slopes may remain.",
                unit="predicted_depth_units")


def object_geometry(mask, depth, intrinsics, valid, plane, config=None):
    cfg = DEFAULTS if config is None else config
    selected = mask & valid
    area = int(mask.sum())
    ratio = float(selected.sum() / max(area, 1))
    if selected.sum() < 30 or ratio < cfg["min_mask_valid_fraction"]:
        return dict(status="failed", reason="insufficient_object_depth", valid_depth_ratio=ratio)
    points, _, _ = project(depth, intrinsics, selected)
    normal, origin = np.asarray(plane["normal_camera"]), np.asarray(plane["origin_camera"])
    axes = np.asarray(plane["axes_camera"])
    center = points.mean(axis=0)
    canonical = axes @ (center - origin)
    height = (points - origin) @ normal
    return dict(status="ok", centroid_camera_xyz=center.tolist(), centroid_ground_xyz=canonical.tolist(),
                z_mean=float(height.mean()), z_median=float(np.median(height)),
                z_p10=float(np.percentile(height, 10)), z_p90=float(np.percentile(height, 90)),
                valid_depth_ratio=ratio, point_count=len(points), unit="predicted_depth_units",
                definition="Mean of visible masked predicted 3D points, not physical extent or complete-object center.")


def build_evidence(depth, intrinsics, valid, masks, prompts, config=None):
    cfg = DEFAULTS if config is None else config
    excluded = masks["subject"] | masks["reference"]
    plane = estimate_ground(depth, intrinsics, valid, excluded, cfg)
    output = dict(version=VERSION, evidence_type="estimated_ground_vertical_3d", ground_estimation=plane,
                  coordinate_frame="estimated_background_plane_z_up", uses_gt_depth=False, uses_gt_pose=False,
                  mask_semantics="union_of_cached_SAM3_masks", objects={})
    if plane["status"] != "ok":
        output.update(status="failed", reason=plane["reason"])
        return output
    for role in ("subject", "reference"):
        output["objects"][role] = dict(prompt=prompts[role], **object_geometry(masks[role], depth, intrinsics, valid, plane, cfg))
    if any(obj["status"] != "ok" for obj in output["objects"].values()):
        output.update(status="failed", reason="invalid_object_3d")
        return output
    a, b = output["objects"]["subject"], output["objects"]["reference"]
    difference = a["z_mean"] - b["z_mean"]
    guard_band = 2 * plane["distance_threshold"]
    output.update(status="ok", reason="estimated_vertical_coordinates_available",
                  comparison=dict(subject_minus_reference_z=difference, ordering="subject_higher" if difference > guard_band else
                                  "reference_higher" if difference < -guard_band else "unresolved_close_centroids",
                                  guard_band=guard_band, unit="predicted_depth_units"))
    if abs(difference) <= guard_band:
        output.update(status="failed", reason="vertical_order_below_plane_resolution")
    return output
