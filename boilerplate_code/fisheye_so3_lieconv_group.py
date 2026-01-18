import math
from boilerplate_code.fisheye_yolo_op_utils import cross_matrix, cosc, norm, sinc, sinc_inv, uncross_matrix
from boilerplate_code.fisheye_camera_model import FisheyeCameraModel
import torch

# How you'd use it with LieConv:
# You typically feed LieConv a tuple (coords, values, mask):
#   • coords: shape [bs, n, 2], pixels ✔️ (the wrapper unprojects them)
#   • values: shape [bs, n, c], pixel values/features
#   • mask: shape [bs, n], bool

# Example: flatten an image to a point cloud:
#
# img: (bs, C, H, W) torch tensor
# bs, C, H, W = img.shape
#
# uu, vv = torch.meshgrid(
#     torch.arange(W, device=img.device),
#     torch.arange(H, device=img.device),
#     indexing="xy",
# )
# p = torch.stack([uu, vv], dim=-1).reshape(1, H*W, 2).repeat(bs, 1, 1).float()
# v = img.permute(0, 2, 3, 1).reshape(bs, H*W, C)
# m = torch.ones(bs, H*W, dtype=torch.bool, device=img.device)
#
# cam = FisheyeCameraModel(fx=430.0, fy=430.0, cx=Wx/2, cy=Hx/2, model="equidistant")
# G = FisheyeSO3(cam)
#
# embedded_locations, expanded_v, expanded_mask = G.lift((p, v, m), nsamples=4)
#
# embedded_locations is exactly what LieConv uses to build an SO(3)-equivariant "convolution neighborhood"
# — but now the geometry comes from your fisheye camera.

# -----------------------------------------------------------------------------
# Why this is the "correct" group (and why it will help)
#
# - Your fisheye warp is *not* a linear transform in pixel space => not a matrix Lie group action on ℝ².
# - But it is exactly the pushforward of SO(3) acting on rays:
#       u' = π(R⋅π⁻¹(u))
# - So the clean way is: **encode pixels as rays**, and run **SO(3)-equivariant kernels**
#
# This gets you your "replace shift-equivariance" goal with a real Lie group.
# -----------------------------------------------------------------------------


# ---------------------------
# LieConv base interface
# ---------------------------

class LieGroup(object):
    rep_dim = NotImplemented
    lie_dim = NotImplemented
    q_dim = NotImplemented

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def exp(self, a):
        raise NotImplementedError

    def log(self, u):
        raise NotImplementedError

    def lifted_elems(self, xyz, nsamples):
        raise NotImplementedError

    def inv(self, g):
        return self.exp(-self.log(g))

    def elems2pairs(self, a):
        vinv = self.exp(-a.unsqueeze(-3))
        u = self.exp(a.unsqueeze(-2))
        return self.log(vinv @ u)

    def lift(self, x, nsamples, **kwargs):
        p, v, m = x
        expanded_a, expanded_q = self.lifted_elems(p, nsamples, **kwargs)

        # v and m expansion to match lifts
        nsamples_eff = expanded_a.shape[-2] // m.shape[-1]
        expanded_v = v[..., None, :].repeat((1,) * len(v.shape[:-1]) + (nsamples_eff, 1))
        expanded_v = expanded_v.reshape(*expanded_a.shape[:-1], v.shape[-1])

        expanded_mask = m[..., None].repeat((1,) * len(m.shape[:-1]) + (nsamples_eff,))
        expanded_mask = expanded_mask.reshape(*expanded_a.shape[:-1])

        paired_a = self.elems2pairs(expanded_a)

        if expanded_q is not None and self.q_dim > 0:
            q_in = expanded_q.unsqueeze(-2).expand(*paired_a.shape[:-1], self.q_dim)
            q_out = expanded_q.unsqueeze(-3).expand(*paired_a.shape[:-1], self.q_dim)
            embedded_locations = torch.cat([paired_a, q_in, q_out], dim=-1)
        else:
            embedded_locations = paired_a

        return (embedded_locations, expanded_v, expanded_mask)


# ---------------------------
# SO(3) action on S^2
# ---------------------------

class SO3OnS2(LieGroup):
    """
    Lie group: SO(3)
    Coordinates: unit sphere directions (rays) in R^3
    - rep_dim = 3 (the linear representation is rotation matrices acting on R^3)
    - lie_dim = 3 (axis-angle / so(3) vector)
    - q_dim = 0 (we're restricting to the unit sphere; no radius quotient)
    """
    lie_dim = 3
    rep_dim = 3
    q_dim = 0

    def __init__(self, alpha=1.0):
        # alpha irrelevant when q_dim=0; set alpha=1.0 to use only group distance
        super().__init__(alpha=alpha)

    def exp(self, w):
        """
        Rodrigues' formula.
        w: (...,3) axis-angle vector (angle = ||w||, axis = w/||w||)
        returns R: (...,3,3)
        """
        theta = norm(w, dim=-1, keepdim=True)[..., None]  # (...,1,1)
        K = cross_matrix(w)                               # (...,3,3)
        I = torch.eye(3, device=w.device, dtype=w.dtype)
        # broadcast I
        while len(I.shape) < len(K.shape):
            I = I.unsqueeze(0)

        R = I + K * sinc(theta) + (K @ K) * cosc(theta)
        return R

    def log(self, R):
        """
        Map R -> axis-angle vector in so(3)
        R: (...,3,3)
        returns w: (...,3)
        """
        trR = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        costheta = torch.clamp((trR - 1.0) / 2.0, min=-1.0, max=1.0).unsqueeze(-1)
        theta = torch.acos(costheta)  # (...,1)

        # For small angles, uncross_matrix(R - R^T)/2 ~ w
        w = uncross_matrix(R) * sinc_inv(theta)
        return w

    def lifted_elems(self, pt, nsamples, **kwargs):
        """
        Lift directions on the unit sphere S^2 (embedded in R^3) to SO(3).

        pt: (...,n,3) rays (do not need to be unit; will normalize)
        nsamples: number of stabilizer samples (rotations around base axis)

        Returns:
          a: (..., n*nsamples, 3)  (so(3) coords)
          q: None
        """
        device, dtype = pt.device, pt.dtype
        *bs, n, _ = pt.shape

        d = pt / norm(pt, dim=-1, keepdim=True)  # (...,n,3)

        # Base direction (north pole): zhat = (0,0,1)
        zhat = torch.zeros(*bs, n, nsamples, 3, device=device, dtype=dtype)
        zhat[..., 2] = 1.0

        # Stabilizer H = SO(2): rotations around zhat
        theta = (2.0 * math.pi) * torch.rand(*bs, n, nsamples, 1, device=device, dtype=dtype)
        Rz = self.exp(zhat * theta)  # (...,n,ns,3,3)

        # Minimal rotation taking zhat -> d
        d_rep = d.unsqueeze(-2).expand(*bs, n, nsamples, 3)
        w = torch.cross(zhat, d_rep, dim=-1)  # axis * sin(angle)
        sin = norm(w, dim=-1)                 # (...,n,ns)
        cos = (zhat * d_rep).sum(dim=-1)      # (...,n,ns)

        angle = torch.atan2(sin, cos).unsqueeze(-1)  # (...,n,ns,1)
        Rp = self.exp(w * sinc_inv(angle))           # (...,n,ns,3,3)

        A = self.log(Rp @ Rz)                        # (...,n,ns,3)

        flat_a = A.reshape(*bs, n * nsamples, 3)
        return flat_a, None

# ---------------------------
# Fisheye "pixel transform" group wrapper
# ---------------------------

class FisheyeSO3(SO3OnS2):
    """
    This is still SO(3) mathematically,
    but it accepts pixel coordinates (...,n,2) and internally unprojects them onto S^2.

    This is the group action that corresponds to:
      u' = pi( R * pi^{-1}(u) )
    """
    def __init__(self, camera: FisheyeCameraModel, alpha=1.0):
        super().__init__(alpha=alpha)
        self.camera = camera

    def lifted_elems(self, pt, nsamples, **kwargs):
        # pt can be either rays (...,n,3) or pixels (...,n,2)
        if pt.shape[-1] == 2:
            pt = self.camera.unproject(pt)  # -> (...,n,3)
        return super().lifted_elems(pt, nsamples, **kwargs)

    @torch.no_grad()
    def warp_pixels(self, uv, R):
        """
        Convenience: apply the induced fisheye warp:
          uv' = project( R @ unproject(uv) )
        uv: (...,2)
        R: (...,3,3) (broadcastable)
        """
        d = self.camera.unproject(uv)         # (...,3)
        d2 = (R @ d.unsqueeze(-1)).squeeze(-1)
        return self.camera.project(d2)
