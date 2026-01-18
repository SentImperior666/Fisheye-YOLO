import torch

# ---------------------------
# Fisheye camera model
# ---------------------------

class FisheyeCameraModel:
    """
    Minimal fisheye unprojection/projection (no distortion poly here),
    suitable for defining the SO(3) action via rays.

    Supports models:
    - equidistant:       r = theta
    - equisolid:         r = 2 sin(theta/2)
    - orthographic:      r = sin(theta)
    - stereographic:     r = 2 tan(theta/2)

    Here r is in "normalized focal units":
      x = (u-cx)/fx
      y = (v-cy)/fy
      r = sqrt(x^2+y^2)
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        model: str = "equidistant",
        flip_y: bool = True,   # image y-down -> convert to y-up (right-handed camera)
        eps: float = 1e-8,
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.model = model.lower()
        self.flip_y = flip_y
        self.eps = eps

        if self.model not in {"equidistant", "equisolid", "orthographic", "stereographic"}:
            raise ValueError(f"Unknown fisheye model: {model}")

    def _theta_from_r(self, r):
        if self.model == "equidistant":
            return r
        if self.model == "equisolid":
            return 2.0 * torch.asin(torch.clamp(r / 2.0, max=1.0 - 1e-7))
        if self.model == "orthographic":
            return torch.asin(torch.clamp(r, max=1.0 - 1e-7))
        if self.model == "stereographic":
            return 2.0 * torch.atan(r / 2.0)
        raise RuntimeError("unreachable")

    def _r_from_theta(self, theta):
        if self.model == "equidistant":
            return theta
        if self.model == "equisolid":
            return 2.0 * torch.sin(theta / 2.0)
        if self.model == "orthographic":
            return torch.sin(theta)
        if self.model == "stereographic":
            return 2.0 * torch.tan(theta / 2.0)
        raise RuntimeError("unreachable")

    def unproject(self, uv):
        """
        uv: (...,2) pixels
        returns rays: (...,3) unit directions in camera frame
        """
        u = uv[..., 0]
        v = uv[..., 1]

        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        if self.flip_y:
            y = -y

        r = torch.sqrt(x * x + y * y + self.eps)
        phi = torch.atan2(y, x)
        theta = self._theta_from_r(r)

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        dx = sin_t * torch.cos(phi)
        dy = sin_t * torch.sin(phi)
        dz = cos_t

        d = torch.stack([dx, dy, dz], dim=-1)
        d = d / norm(d, dim=-1, keepdim=True)
        return d

    def project(self, d):
        """
        d: (...,3) unit directions
        returns uv: (...,2) pixels
        """
        d = d / norm(d, dim=-1, keepdim=True)

        dz = torch.clamp(d[..., 2], -1.0, 1.0)
        theta = torch.acos(dz)
        phi = torch.atan2(d[..., 1], d[..., 0])

        r = self._r_from_theta(theta)

        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        if self.flip_y:
            y = -y

        u = x * self.fx + self.cx
        v = y * self.fy + self.cy
        return torch.stack([u, v], dim=-1)