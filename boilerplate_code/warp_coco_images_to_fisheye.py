import numpy as np
import cv2

# === Fisheye synthetic image generation equations ===
# Output fisheye camera (virtual):
#   - output size: (H_out, W_out)
#   - principal point: (cx, cy) (usually image center)
#   - fisheye FOV (e.g. 180° or 160°)
#   - fisheye projection, equidistant: r = f * theta
#
# To cover a max fisheye angle theta_max in image radius R:
#     f_fish = R / theta_max          (theta in radians)
#
# Input pinhole (source) camera:
#   - given image width W and assumed source FOV (rectilinear, e.g. 70-90°):
#     f_src = (W/2) / tan(FOV_src/2)   (FOV in radians)


# === How to warp an image properly (backward mapping) ===
# For each output fisheye pixel (u, v):
#   1. unproject it  -- ray direction d ∈ S^2 using the fisheye model
#   2. project that ray into the source pinhole image -- (u_s, v_s)
#   3. sample source with interpolation (cv2.remap)
# This is "render normal image as seen through fisheye lens".


def fisheye_unproject_equdist(u, v, fx, fy, cx, cy):
    """
    Equidistant fisheye: r = f * theta
    Returns ray direction d=(dx,dy,dz) in camera frame for each pixel.
    u,v can be numpy arrays.
    """
    x = (u - cx) / fx
    y = (v - cy) / fy

    r = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)

    theta = r  # equidistant: r = theta if fx=fy=f in normalized coords
    # but since we used fx/fy in pixel units, theta is already in radians if x,y are normalized by f

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    dx = sin_t * np.cos(phi)
    dy = sin_t * np.sin(phi)
    dz = cos_t
    return dx, dy, dz

def pinhole_project(dx, dy, dz, fx, fy, cx, cy):
    """
    Standard pinhole projection: x = dx/dz, y=dy/dz
    """
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
    """
    Takes a normal (rectilinear/pinhole-like) image and renders it as equidistant fisheye.
    """
    Hs, Ws = img_bgr.shape[:2]
    Ho, Wo = out_size

    # --- Output fisheye intrinsics ---
    cx_o, cy_o = (Wo - 1) / 2.0, (Ho - 1) / 2.0
    R = min(cx_o, cy_o)
    theta_max = np.deg2rad(fov_fisheye_deg / 2.0)
    f_fish = R / theta_max
    fx_o = fy_o = f_fish

    # --- Source pinhole intrinsics ---
    cx_s, cy_s = (Ws - 1) / 2.0, (Hs - 1) / 2.0
    f_src = (Ws / 2.0) / np.tan(np.deg2rad(fov_src_deg / 2.0))
    fx_s = fy_s = f_src

    # --- Build output grid ---
    uo, vo = np.meshgrid(np.arange(Wo, dtype=np.float32),
                        np.arange(Ho, dtype=np.float32))

    # --- Unproject fisheye pixel -> ray ---
    dx, dy, dz = fisheye_unproject_equdist(uo, vo, fx_o, fy_o, cx_o, cy_o)

    # --- Optional: mask outside fisheye circle (because theta beyond theta_max is invalid) ---
    # Here theta ~= r_norm = sqrt(x^2+y^2) = sqrt(((u-cx)/f)^2+...)
    r_norm = np.sqrt(((uo - cx_o) / fx_o)**2 + ((vo - cy_o) / fy_o)**2)
    valid = r_norm <= theta_max + 1e-6  # rays inside chosen FOV

    # --- Project ray -> source pinhole pixel ---
    us, vs = pinhole_project(dx, dy, dz, fx_s, fy_s, cx_s, cy_s)

    # invalid areas: map outside image
    us[~valid] = -1
    vs[~valid] = -1

    # --- Remap ---
    map_x = us.astype(np.float32)
    map_y = vs.astype(np.float32)

    out = cv2.remap(
        img_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return out

if __name__ == "__main__":
    img = cv2.imread("input.jpg")
    fisheye = warp_pinhole_to_fisheye(img, out_size=(1024, 1024), fov_fisheye_deg=180, fov_src_deg=90)
    cv2.imwrite("fisheye.jpg", fisheye)
