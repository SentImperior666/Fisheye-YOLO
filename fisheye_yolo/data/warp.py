import numpy as np
import cv2

# Supported fisheye projection models
FISHEYE_MODELS = ("equidistant", "equisolid", "stereographic", "orthographic")


def _theta_from_r(r, model):
    """Convert normalized radius to angle theta based on projection model."""
    if model == "equidistant":
        return r
    elif model == "equisolid":
        return 2.0 * np.arcsin(np.clip(r / 2.0, -1.0 + 1e-7, 1.0 - 1e-7))
    elif model == "orthographic":
        return np.arcsin(np.clip(r, -1.0 + 1e-7, 1.0 - 1e-7))
    elif model == "stereographic":
        return 2.0 * np.arctan(r / 2.0)
    else:
        raise ValueError(f"Unknown fisheye model: {model}")


def _r_from_theta(theta, model):
    """Convert angle theta to normalized radius based on projection model."""
    if model == "equidistant":
        return theta
    elif model == "equisolid":
        return 2.0 * np.sin(theta / 2.0)
    elif model == "orthographic":
        return np.sin(theta)
    elif model == "stereographic":
        return 2.0 * np.tan(theta / 2.0)
    else:
        raise ValueError(f"Unknown fisheye model: {model}")


def _max_r_for_model(theta_max, model):
    """Get the maximum normalized radius for a given theta_max and model."""
    return _r_from_theta(theta_max, model)


def intrinsics_from_fov(w, h, fov_deg):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    f = (w / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    return f, f, cx, cy


def fisheye_intrinsics_equdist(w, h, fov_fisheye_deg):
    """Legacy function for backwards compatibility."""
    return fisheye_intrinsics(w, h, fov_fisheye_deg, model="equidistant")


def fisheye_intrinsics(w, h, fov_fisheye_deg, model="equidistant"):
    """
    Compute fisheye camera intrinsics for a given image size and FOV.
    
    Returns fx, fy, cx, cy, theta_max where the focal length is computed
    such that the maximum angle maps to the image radius.
    """
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    radius = min(cx, cy)
    theta_max = np.deg2rad(fov_fisheye_deg / 2.0)
    # For the given model, find f such that r_max * f = radius
    r_max = _max_r_for_model(theta_max, model)
    f = radius / r_max if r_max > 1e-9 else radius
    return f, f, cx, cy, theta_max


def fisheye_unproject_equdist(u, v, fx, fy, cx, cy):
    """Legacy function for backwards compatibility."""
    return fisheye_unproject(u, v, fx, fy, cx, cy, model="equidistant")


def fisheye_unproject(u, v, fx, fy, cx, cy, model="equidistant"):
    """
    Unproject fisheye pixel coordinates to 3D ray directions.
    
    Supports models: equidistant, equisolid, stereographic, orthographic.
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    theta = _theta_from_r(r, model)
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
    fisheye_model="equidistant",
    interpolation=cv2.INTER_LANCZOS4,
    full_frame=False,
    sharpen=0.0,
):
    """
    Warp a pinhole camera image to a fisheye projection.
    
    Args:
        img_bgr: Input BGR image from a pinhole camera.
        out_size: Output image size (height, width).
        fov_fisheye_deg: Field of view of the output fisheye image in degrees.
        fov_src_deg: Field of view of the source pinhole image in degrees.
        border_value: Value to use for pixels outside the source image.
        fisheye_model: Fisheye projection model. One of:
            - "equidistant": r = f * theta (linear angle-to-radius)
            - "equisolid": r = 2f * sin(theta/2) (preserves relative areas)
            - "stereographic": r = 2f * tan(theta/2) (preserves angles)
            - "orthographic": r = f * sin(theta) (max compression at edges)
        interpolation: OpenCV interpolation method. Options:
            - cv2.INTER_LANCZOS4: Highest quality (default)
            - cv2.INTER_CUBIC: High quality, faster
            - cv2.INTER_LINEAR: Fast, lower quality
        full_frame: If True, fill the entire rectangular output (no circular mask).
            This produces images similar to real full-frame fisheye cameras.
            If False (default), creates circular fisheye with black borders.
        sharpen: Sharpening strength (0.0 = none, 0.5 = moderate, 1.0 = strong).
            Helps counteract blur from stretching in full-frame mode.
    
    Returns:
        Warped fisheye image.
    """
    if fisheye_model not in FISHEYE_MODELS:
        raise ValueError(f"Unknown fisheye model: {fisheye_model}. Must be one of {FISHEYE_MODELS}")
    
    h_s, w_s = img_bgr.shape[:2]
    h_o, w_o = out_size

    cx_o, cy_o = (w_o - 1) / 2.0, (h_o - 1) / 2.0
    theta_max = np.deg2rad(fov_fisheye_deg / 2.0)

    if full_frame:
        # Full-frame mode: fisheye fills entire rectangle
        # Compute focal length so that the image corners correspond to theta_max
        # Distance from center to corner in pixels
        corner_dist = np.sqrt(cx_o**2 + cy_o**2)
        r_max = _max_r_for_model(theta_max, fisheye_model)
        fx_o = fy_o = corner_dist / r_max if r_max > 1e-9 else corner_dist
    else:
        # Circular mode: fisheye inscribed in rectangle
        radius = min(cx_o, cy_o)
        r_max = _max_r_for_model(theta_max, fisheye_model)
        fx_o = fy_o = radius / r_max if r_max > 1e-9 else radius

    # Compute pinhole intrinsics for the source image
    cx_s, cy_s = (w_s - 1) / 2.0, (h_s - 1) / 2.0
    f_src = (w_s / 2.0) / np.tan(np.deg2rad(fov_src_deg / 2.0))
    fx_s = fy_s = f_src

    # Create output pixel grid
    uo, vo = np.meshgrid(np.arange(w_o, dtype=np.float32), np.arange(h_o, dtype=np.float32))
    
    # Unproject fisheye pixels to rays
    dx, dy, dz = fisheye_unproject(uo, vo, fx_o, fy_o, cx_o, cy_o, model=fisheye_model)
    
    # Project rays to pinhole source coordinates
    us, vs = pinhole_project(dx, dy, dz, fx_s, fy_s, cx_s, cy_s)

    if not full_frame:
        # Compute validity mask (within FOV) - only for circular mode
        x_norm = (uo - cx_o) / fx_o
        y_norm = (vo - cy_o) / fy_o
        r_norm = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        r_max = _max_r_for_model(theta_max, fisheye_model)
        valid = r_norm <= r_max + 1e-6
        us[~valid] = -1
        vs[~valid] = -1

    out = cv2.remap(
        img_bgr,
        us.astype(np.float32),
        vs.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    
    # Apply sharpening if requested
    if sharpen > 0:
        # Unsharp mask: out + sharpen * (out - blur)
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
        out = cv2.addWeighted(out, 1.0 + sharpen, blur, -sharpen, 0)
    
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
    fisheye_model="equidistant",
):
    """
    Warp a bounding box from pinhole to fisheye coordinates.
    
    Args:
        bbox_xyxy: Bounding box in (x1, y1, x2, y2) format.
        src_size: Source image size (height, width).
        dst_size: Destination fisheye image size (height, width).
        fov_src_deg: Source pinhole FOV in degrees.
        fov_fisheye_deg: Destination fisheye FOV in degrees.
        samples_per_edge: Number of samples per bbox edge for warping.
        clip: Whether to clip the result to image bounds.
        min_box_size: Minimum box size to return (otherwise None).
        fisheye_model: Fisheye projection model.
    
    Returns:
        Warped bounding box (x1, y1, x2, y2) or None if invalid.
    """
    h_s, w_s = src_size
    h_d, w_d = dst_size
    x1, y1, x2, y2 = bbox_xyxy

    fx_s, fy_s, cx_s, cy_s = intrinsics_from_fov(w_s, h_s, fov_src_deg)
    fx_f, fy_f, cx_f, cy_f, theta_max = fisheye_intrinsics(w_d, h_d, fov_fisheye_deg, model=fisheye_model)

    pts = sample_bbox_boundary_points_xyxy(x1, y1, x2, y2, samples_per_edge=samples_per_edge)
    u = pts[:, 0]
    v = pts[:, 1]
    d = pinhole_pixel_to_ray(u, v, fx_s, fy_s, cx_s, cy_s)
    uf, vf, valid = ray_to_fisheye(d, fx_f, fy_f, cx_f, cy_f, theta_max=theta_max, model=fisheye_model)

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
    """Legacy function for backwards compatibility."""
    return ray_to_fisheye(d, fx, fy, cx, cy, theta_max=theta_max, model="equidistant")


def ray_to_fisheye(d, fx, fy, cx, cy, theta_max=None, model="equidistant"):
    """
    Project 3D ray directions to fisheye pixel coordinates.
    
    Args:
        d: Ray directions (..., 3)
        fx, fy, cx, cy: Fisheye intrinsics
        theta_max: Maximum angle (for validity check)
        model: Fisheye projection model
    
    Returns:
        u, v: Pixel coordinates
        valid: Boolean mask for valid projections
    """
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    dz = np.clip(dz, -1.0, 1.0)
    theta = np.arccos(dz)
    phi = np.arctan2(dy, dx)
    r_norm = _r_from_theta(theta, model)
    r = fx * r_norm  # Scale by focal length
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
