import math

import torch

from fisheye_yolo.utils.third_party import ensure_lieconv_on_path

ensure_lieconv_on_path()

from lie_conv.lieGroups import LieGroup

from fisheye_yolo.geometry.fisheye_camera import FisheyeCameraModel
from fisheye_yolo.geometry.lieconv_utils import cosc, cross_matrix, norm, sinc, sinc_inv, uncross_matrix


class SO3OnS2(LieGroup):
    """
    Lie group: SO(3)
    Coordinates: unit sphere directions (rays) in R^3
    - rep_dim = 3 (the linear representation is rotation matrices acting on R^3)
    - lie_dim = 3 (axis-angle / so(3) vector)
    - q_dim = 0 (restrict to the unit sphere; no radius quotient)
    """

    lie_dim = 3
    rep_dim = 3
    q_dim = 0

    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)

    def exp(self, w):
        """
        Rodrigues' formula.
        w: (...,3) axis-angle vector (angle = ||w||, axis = w/||w||)
        returns R: (...,3,3)
        """
        theta = norm(w, dim=-1, keepdim=True)[..., None]
        K = cross_matrix(w)
        I = torch.eye(3, device=w.device, dtype=w.dtype)
        while len(I.shape) < len(K.shape):
            I = I.unsqueeze(0)
        return I + K * sinc(theta) + (K @ K) * cosc(theta)

    def log(self, R):
        """
        Map R -> axis-angle vector in so(3)
        R: (...,3,3)
        returns w: (...,3)
        """
        trR = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        costheta = torch.clamp((trR - 1.0) / 2.0, min=-1.0, max=1.0).unsqueeze(-1)
        theta = torch.acos(costheta)
        return uncross_matrix(R) * sinc_inv(theta)

    def components2matrix(self, a):
        return cross_matrix(a)

    def matrix2components(self, A):
        return uncross_matrix(A)

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

        d = pt / norm(pt, dim=-1, keepdim=True)

        zhat = torch.zeros(*bs, n, nsamples, 3, device=device, dtype=dtype)
        zhat[..., 2] = 1.0

        theta = (2.0 * math.pi) * torch.rand(*bs, n, nsamples, 1, device=device, dtype=dtype)
        Rz = self.exp(zhat * theta)

        d_rep = d.unsqueeze(-2).expand(*bs, n, nsamples, 3)
        w = torch.cross(zhat, d_rep, dim=-1)
        sin = norm(w, dim=-1)
        cos = (zhat * d_rep).sum(dim=-1)

        angle = torch.atan2(sin, cos).unsqueeze(-1)
        Rp = self.exp(w * sinc_inv(angle))

        A = self.log(Rp @ Rz)
        flat_a = A.reshape(*bs, n * nsamples, 3)
        return flat_a, None


class FisheyeSO3(SO3OnS2):
    """
    SO(3) group wrapper that accepts pixel coordinates (...,n,2)
    and internally unprojects them to S^2 directions.

    This matches the induced fisheye warp:
      u' = pi( R * pi^{-1}(u) )
    """

    def __init__(self, camera: FisheyeCameraModel, alpha=1.0):
        super().__init__(alpha=alpha)
        self.camera = camera

    def lifted_elems(self, pt, nsamples, **kwargs):
        if pt.shape[-1] == 2:
            pt = self.camera.unproject(pt)
        return super().lifted_elems(pt, nsamples, **kwargs)

    @torch.no_grad()
    def warp_pixels(self, uv, R):
        """
        Apply the induced fisheye warp:
          uv' = project( R @ unproject(uv) )
        """
        d = self.camera.unproject(uv)
        d2 = (R @ d.unsqueeze(-1)).squeeze(-1)
        return self.camera.project(d2)
