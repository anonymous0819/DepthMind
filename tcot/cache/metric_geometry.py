from __future__ import annotations
import numpy as np

TWO_TARGET_TASKS = {
    'distance_data', 'vertical_distance_data', 'horizontal_distance_data',
}

SKY_DEPTH_THRESHOLD = 1e4

MIN_VALID_DEPTH = 0.5

MIN_MASK_PIXELS = 50

def rescale_K(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    sx = dst_w / src_w
    sy = dst_h / src_h
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2

def resize_depth_to(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a HxW depth map (in meters) to target_hw."""
    from PIL import Image as _Im
    h_t, w_t = target_hw
    if depth.shape == (h_t, w_t):
        return depth
    img = _Im.fromarray(depth.astype(np.float32), mode='F')
    img = img.resize((w_t, h_t), _Im.BILINEAR)
    return np.array(img, dtype=np.float32)

def load_mask_from_npz(path: str) -> np.ndarray | None:
    """Return a (H, W) bool mask.  If the npz stores (N_masks, H, W), pick
    the union (any) of all masks - the sam3 pipeline stores best masks so N
    is usually 1."""
    if not path:
        return None
    try:
        d = np.load(path)
    except Exception:
        return None
    key = 'masks' if 'masks' in d.files else d.files[0]
    m = d[key]
    if m.ndim == 3:
        m = np.any(m.astype(bool), axis=0)
    else:
        m = m.astype(bool)
    return m

def resize_mask_to(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of a bool mask to target_hw."""
    from PIL import Image as _Im
    h_t, w_t = target_hw
    if mask.shape == (h_t, w_t):
        return mask
    img = _Im.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w_t, h_t), _Im.NEAREST)
    return np.array(img, dtype=np.uint8) > 0

def backproject_mask_with_valid(
    mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as ``backproject_mask`` but also returns the boolean ``valid``
    map (H, W) recording which pixels were actually used for backprojection.
    Confidence statistics should be aggregated on this exact set of pixels
    so that they correspond 1:1 to what fed the geometric prediction.
    """
    H, W = depth.shape
    assert mask.shape == depth.shape, f'mask {mask.shape} vs depth {depth.shape}'
    valid = mask & (depth > MIN_VALID_DEPTH) & (depth < SKY_DEPTH_THRESHOLD)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64), valid
    z = depth[ys, xs].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=-1), valid

def filter_outliers(points: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal using open3d's kNN version, identical to
    the GT pipeline in processor/process_pointcloud.py:
        pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    For each point compute the mean distance to its k nearest neighbours;
    then keep points whose mean-kNN-distance is within std_ratio * sigma of
    the global mean.  This is applied to BOTH GT and estimated depth to keep
    the evaluation protocol identical across depth sources (SAM mask leaks
    and depth-edge overshoots both benefit from it).
    """
    if len(points) < max(k + 1, 10):
        return points
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        _, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=int(k),
                                                    std_ratio=float(std_ratio))
    except Exception:
        return points
    if not keep_idx:
        return points
    return points[np.asarray(keep_idx, dtype=np.int64)]

def compute_physical_value(
    task: str,
    pts_A: np.ndarray,
    pts_B: np.ndarray | None,
) -> float | None:
    """All quantities are computed directly in the CAMERA frame — no pose
    canonicalisation is applied because O3DVQA's ground truth stores every
    numeric answer in raw camera coordinates (see visible_objs/*.json:
    ``cam_obj_loc``, ``cam_bbox_3d``, ``ego_range``).

    Camera convention (verified against ``cam_obj_loc`` vs ``ego_range``):
      * x_cam : right
      * y_cam : down            <- "vertical" component
      * z_cam : forward         <- part of "horizontal" component
      * distance2agent            == ||cam_center||
      * vertical_distance2agent   == |y_cam_center|
      * horizontal_distance2agent == hypot(x_cam_center, z_cam_center)

    Object-object variants (``*_data``) reuse the AABB centre of each mask's
    point cloud (matches vqasynth's ``AABB.get_center()`` semantics), whereas
    the ``*2agent`` variants use the point-cloud mean (matches ``pcd.get_center``).
    Width / height / volume come from ``max - min`` of the AABB along the
    camera axes: width = dx, height = dy (vertical), volume = dx*dy*dz.
    """
    if len(pts_A) < MIN_MASK_PIXELS:
        return None
    if task in TWO_TARGET_TASKS:
        if pts_B is None or len(pts_B) < MIN_MASK_PIXELS:
            return None

    # Agent-relative tasks: cam-frame point-cloud MEAN centre.
    # (Verified against O3DVQA visible_objs/*.json cam_obj_loc:
    #    distance2agent            == ||cam_center||
    #    vertical_distance2agent   == |y_cam_center|
    #    horizontal_distance2agent == |x_cam_center|
    # )
    if task == 'distance2agent':
        cA = pts_A.mean(axis=0)
        return float(np.linalg.norm(cA))
    if task == 'horizontal_distance2agent':
        cA = pts_A.mean(axis=0)
        return float(abs(cA[0]))
    if task == 'vertical_distance2agent':
        cA = pts_A.mean(axis=0)
        return float(abs(cA[1]))

    # Pairwise *_data tasks: cam-frame AABB centre = (min+max)/2 per axis,
    # then take the same axis convention as the *2agent counterparts.
    if task in ('distance_data', 'horizontal_distance_data', 'vertical_distance_data'):
        cA = 0.5 * (pts_A.min(axis=0) + pts_A.max(axis=0))
        cB = 0.5 * (pts_B.min(axis=0) + pts_B.max(axis=0))
        d = cA - cB
        if task == 'distance_data':
            return float(np.linalg.norm(d))
        if task == 'horizontal_distance_data':
            return float(abs(d[0]))
        return float(abs(d[1]))

    # Single-target AABB extent measures (cam-frame axes).
    lo = pts_A.min(axis=0)
    hi = pts_A.max(axis=0)
    dx, dy, dz = hi - lo
    if task == 'width_data':
        return float(dx)              # left-right span in cam frame
    if task == 'height_data':
        return float(dy)              # vertical span in cam frame
    if task == 'volume_data':
        return float(dx * dy * dz)
    raise ValueError(f'unknown task: {task}')
