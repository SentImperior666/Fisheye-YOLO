import numpy as np

# -----------------------------
# Intrinsics helpers
# -----------------------------

def intrinsics_from_fov(W, H, fov_deg):
    """
    Create "standard" pinhole intrinsics from a chosen horizontal FOV.
    """
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    f = (W / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    return f, f, cx, cy  # fx, fy, cx, cy

def fisheye_intrinsics_equdist(W, H, fov_fisheye_deg):
    """
    Equidistant fisheye r = f * theta.
    We choose f so that theta_max = fov/2 hits the image radius.
    """
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    R = min(cx, cy)
    theta_max = np.deg2rad(fov_fisheye_deg / 2.0)
    f = R / theta_max
    return f, f, cx, cy, theta_max


# -----------------------------
# Pinhole pixel -> ray
# -----------------------------

def pinhole_pixel_to_ray(u, v, fx, fy, cx, cy):
    """
    u,v: (...,) pixel coords in source image
    returns unit ray d=(dx,dy,dz)
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)

    d = np.stack([x, y, z], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
    return d


# -----------------------------
# Ray -> equidistant fisheye pixel
# -----------------------------

def ray_to_fisheye_equdist(d, fx, fy, cx, cy, theta_max=None):
    """
    d: (...,3) unit ray
    equidistant: r = f * theta, theta = acos(dz), phi = atan2(dy,dx)
    """
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    dz = np.clip(dz, -1.0, 1.0)

    theta = np.arccos(dz)
    phi = np.arctan2(dy, dx)

    r = fx * theta  # fx==fy
    u = cx + r * np.cos(phi)
    v = cy + r * np.sin(phi)

    valid = np.ones_like(u, dtype=bool)
    if theta_max is not None:
        valid &= (theta <= theta_max + 1e-6)

    return u, v, valid


# -----------------------------
# Sampling bbox boundary points
# -----------------------------

def sample_bbox_boundary_points_xyxy(x1, y1, x2, y2, samples_per_edge=9):
    """
    Sample points along bbox boundary so distortion doesn't shrink the box.
    Returns (N,2) points.
    """
    xs = np.linspace(x1, x2, samples_per_edge)
    ys = np.linspace(y1, y2, samples_per_edge)

    top    = np.stack([xs, np.full_like(xs, y1)], axis=-1)
    bottom = np.stack([xs, np.full_like(xs, y2)], axis=-1)
    left   = np.stack([np.full_like(ys, x1), ys], axis=-1)
    right  = np.stack([np.full_like(ys, x2), ys], axis=-1)

    pts = np.concatenate([top, bottom, left, right], axis=0)
    return pts


# -----------------------------
# Warp a single bbox (xyxy)
# -----------------------------

def warp_bbox_xyxy_pinhole_to_fisheye(
    bbox_xyxy,
    src_size,         # (Hs, Ws)
    dst_size,         # (Hd, Wd)
    fov_src_deg=90.0,
    fov_fisheye_deg=180.0,
    samples_per_edge=9,
    clip=True,
    min_box_size=2.0,
):
    """
    bbox_xyxy: (x1,y1,x2,y2) in source pixels
    Returns:
      bbox_xyxy_dst or None if fully invalid
    """
    Hs, Ws = src_size
    Hd, Wd = dst_size

    x1, y1, x2, y2 = bbox_xyxy

    # build intrinsics
    fx_s, fy_s, cx_s, cy_s = intrinsics_from_fov(Ws, Hs, fov_src_deg)
    fx_f, fy_f, cx_f, cy_f, theta_max = fisheye_intrinsics_equdist(Wd, Hd, fov_fisheye_deg)

    # sample boundary points
    pts = sample_bbox_boundary_points_xyxy(x1, y1, x2, y2, samples_per_edge=samples_per_edge)
    u = pts[:, 0]
    v = pts[:, 1]

    # project source pixel -> ray
    d = pinhole_pixel_to_ray(u, v, fx_s, fy_s, cx_s, cy_s)

    # ray -> fisheye pixel
    uf, vf, valid = ray_to_fisheye_equdist(d, fx_f, fy_f, cx_f, cy_f, theta_max=theta_max)

    if valid.sum() < 4:
        # too few valid points => object mostly outside fisheye view
        return None

    uf = uf[valid]
    vf = vf[valid]

    x1n, y1n = float(np.min(uf)), float(np.min(vf))
    x2n, y2n = float(np.max(uf)), float(np.max(vf))

    if clip:
        x1n = max(0.0, min(x1n, Wd - 1.0))
        x2n = max(0.0, min(x2n, Wd - 1.0))
        y1n = max(0.0, min(y1n, Hd - 1.0))
        y2n = max(0.0, min(y2n, Hd - 1.0))

    if (x2n - x1n) < min_box_size or (y2n - y1n) < min_box_size:
        return None

    return (x1n, y1n, x2n, y2n)


# -----------------------------
# Convert warped bbox to YOLO
# -----------------------------

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    xc = (x1 + x2) / 2.0 / W
    yc = (y1 + y2) / 2.0 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h

if __name__ == "__main__":
    src_H, src_W = 480, 640
    dst_H, dst_W = 640, 640

    bbox_src = (100, 120, 220, 260)  # xyxy in source

    bbox_dst = warp_bbox_xyxy_pinhole_to_fisheye(
        bbox_src,
        src_size=(src_H, src_W),
        dst_size=(dst_H, dst_W),
        fov_src_deg=90,
        fov_fisheye_deg=180,
        samples_per_edge=11,
    )

    if bbox_dst is not None:
        x1, y1, x2, y2 = bbox_dst
        xc, yc, w, h = xyxy_to_yolo(x1, y1, x2, y2, dst_W, dst_H)
        print("YOLO:", xc, yc, w, h)
    else:
        print("bbox dropped (outside fisheye / too small)")
