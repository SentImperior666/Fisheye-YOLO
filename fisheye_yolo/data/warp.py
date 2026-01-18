import numpy as np
import cv2


def intrinsics_from_fov(w, h, fov_deg):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    f = (w / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    return f, f, cx, cy


def fisheye_intrinsics_equdist(w, h, fov_fisheye_deg):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    radius = min(cx, cy)
    theta_max = np.deg2rad(fov_fisheye_deg / 2.0)
    f = radius / theta_max
    return f, f, cx, cy, theta_max


def fisheye_unproject_equdist(u, v, fx, fy, cx, cy):
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    theta = r
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    dx = sin_t * np.cos(phi)
    dy = sin_t * np.sin(phi)
    dz = cos_t
    return dx, dy, dz


def pinhole_project(dx, dy, dz, fx, fy, cx, cy):
    eps = 1e-8
    z = np.maximum(dz, eps)
    x = dx / z
    y = dy / z
    u = fx * x + cx
    v = fy * y + cy
    return u, v


def warp_pinhole_to_fisheye(
    img_bgr,
    out_size=(640, 640),
    fov_fisheye_deg=180.0,
    fov_src_deg=90.0,
    border_value=(0, 0, 0),
):
    h_s, w_s = img_bgr.shape[:2]
    h_o, w_o = out_size

    cx_o, cy_o = (w_o - 1) / 2.0, (h_o - 1) / 2.0
    radius = min(cx_o, cy_o)
    theta_max = np.deg2rad(fov_fisheye_deg / 2.0)
    f_fish = radius / theta_max
    fx_o = fy_o = f_fish

    cx_s, cy_s = (w_s - 1) / 2.0, (h_s - 1) / 2.0
    f_src = (w_s / 2.0) / np.tan(np.deg2rad(fov_src_deg / 2.0))
    fx_s = fy_s = f_src

    uo, vo = np.meshgrid(np.arange(w_o, dtype=np.float32), np.arange(h_o, dtype=np.float32))
    dx, dy, dz = fisheye_unproject_equdist(uo, vo, fx_o, fy_o, cx_o, cy_o)
    r_norm = np.sqrt(((uo - cx_o) / fx_o) ** 2 + ((vo - cy_o) / fy_o) ** 2)
    valid = r_norm <= theta_max + 1e-6
    us, vs = pinhole_project(dx, dy, dz, fx_s, fy_s, cx_s, cy_s)
    us[~valid] = -1
    vs[~valid] = -1

    out = cv2.remap(
        img_bgr,
        us.astype(np.float32),
        vs.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return out


def sample_bbox_boundary_points_xyxy(x1, y1, x2, y2, samples_per_edge=9):
    xs = np.linspace(x1, x2, samples_per_edge)
    ys = np.linspace(y1, y2, samples_per_edge)
    top = np.stack([xs, np.full_like(xs, y1)], axis=-1)
    bottom = np.stack([xs, np.full_like(xs, y2)], axis=-1)
    left = np.stack([np.full_like(ys, x1), ys], axis=-1)
    right = np.stack([np.full_like(ys, x2), ys], axis=-1)
    return np.concatenate([top, bottom, left, right], axis=0)


def warp_bbox_xyxy_pinhole_to_fisheye(
    bbox_xyxy,
    src_size,
    dst_size,
    fov_src_deg=90.0,
    fov_fisheye_deg=180.0,
    samples_per_edge=9,
    clip=True,
    min_box_size=2.0,
):
    h_s, w_s = src_size
    h_d, w_d = dst_size
    x1, y1, x2, y2 = bbox_xyxy

    fx_s, fy_s, cx_s, cy_s = intrinsics_from_fov(w_s, h_s, fov_src_deg)
    fx_f, fy_f, cx_f, cy_f, theta_max = fisheye_intrinsics_equdist(w_d, h_d, fov_fisheye_deg)

    pts = sample_bbox_boundary_points_xyxy(x1, y1, x2, y2, samples_per_edge=samples_per_edge)
    u = pts[:, 0]
    v = pts[:, 1]
    d = pinhole_pixel_to_ray(u, v, fx_s, fy_s, cx_s, cy_s)
    uf, vf, valid = ray_to_fisheye_equdist(d, fx_f, fy_f, cx_f, cy_f, theta_max=theta_max)

    if valid.sum() < 4:
        return None

    uf = uf[valid]
    vf = vf[valid]
    x1n, y1n = float(np.min(uf)), float(np.min(vf))
    x2n, y2n = float(np.max(uf)), float(np.max(vf))

    if clip:
        x1n = max(0.0, min(x1n, w_d - 1.0))
        x2n = max(0.0, min(x2n, w_d - 1.0))
        y1n = max(0.0, min(y1n, h_d - 1.0))
        y2n = max(0.0, min(y2n, h_d - 1.0))

    if (x2n - x1n) < min_box_size or (y2n - y1n) < min_box_size:
        return None

    return (x1n, y1n, x2n, y2n)


def pinhole_pixel_to_ray(u, v, fx, fy, cx, cy):
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    d = np.stack([x, y, z], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
    return d


def ray_to_fisheye_equdist(d, fx, fy, cx, cy, theta_max=None):
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    dz = np.clip(dz, -1.0, 1.0)
    theta = np.arccos(dz)
    phi = np.arctan2(dy, dx)
    r = fx * theta
    u = cx + r * np.cos(phi)
    v = cy + r * np.sin(phi)
    valid = np.ones_like(u, dtype=bool)
    if theta_max is not None:
        valid &= theta <= theta_max + 1e-6
    return u, v, valid


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh
