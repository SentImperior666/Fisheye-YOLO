import torch

# ---------------------------
# Utils (minimal LieConv-like)
# ---------------------------

def norm(x, dim=-1, keepdim=False, eps=1e-8):
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))

def sinc(x, eps=1e-6):
    # sin(x)/x stable
    x2 = x * x
    return torch.where(
        x.abs() < eps,
        1 - x2 / 6 + x2 * x2 / 120,
        torch.sin(x) / x,
    )

def sinc_inv(x, eps=1e-6):
    # x/sin(x) stable
    x2 = x * x
    return torch.where(
        x.abs() < eps,
        1 + x2 / 6 + 7 * x2 * x2 / 360,
        x / torch.sin(x),
    )

def cosc(x, eps=1e-6):
    # (1 - cos(x)) / x^2 stable
    x2 = x * x
    return torch.where(
        x.abs() < eps,
        0.5 - x2 / 24 + x2 * x2 / 720,
        (1 - torch.cos(x)) / x2,
    )

def cross_matrix(w):
    """
    w: (...,3)
    returns skew-symmetric matrix K: (...,3,3) such that K @ v = w x v
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    K = torch.stack(
        [
            torch.stack([O, -wz, wy], dim=-1),
            torch.stack([wz, O, -wx], dim=-1),
            torch.stack([-wy, wx, O], dim=-1),
        ],
        dim=-2,
    )
    return K

def uncross_matrix(A):
    """
    A: (...,3,3) assumed skew-symmetric
    returns w: (...,3)
    """
    return torch.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], dim=-1)