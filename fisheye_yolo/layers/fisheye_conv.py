from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fisheye_yolo.geometry.fisheye_camera import FisheyeCameraModel
from fisheye_yolo.utils.third_party import ensure_lieconv_on_path

ensure_lieconv_on_path()

from lie_conv.lieConv import LieConv
from lie_conv.lieGroups import FisheyeSO3


def _pixel_grid(h, w, device, dtype):
    u = torch.arange(w, device=device, dtype=dtype)
    v = torch.arange(h, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([uu, vv], dim=-1).reshape(-1, 2)


class FisheyeLieConv2d(LieConv):
    def __init__(
        self,
        c1,
        c2,
        stride=1,
        camera: Optional[FisheyeCameraModel] = None,
        group: Optional[FisheyeSO3] = None,
        liftsamples=4,
        mc_samples=32,
        fill=1 / 4,
        knn=True,
    ):
        if group is None:
            if camera is None:
                raise ValueError("camera or group is required for FisheyeLieConv2d")
            group = FisheyeSO3(camera)
        super().__init__(
            c1,
            c2,
            mc_samples=mc_samples,
            ds_frac=1,
            group=group,
            fill=fill,
            knn=knn,
            bn=False,
            act="swish",
            mean=False,
        )
        self.group = group
        self.liftsamples = liftsamples
        self.stride = stride

    def forward(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        bs, c, h, w = x.shape
        coords = _pixel_grid(h, w, x.device, x.dtype).unsqueeze(0).repeat(bs, 1, 1)
        values = x.permute(0, 2, 3, 1).reshape(bs, -1, c)
        mask = torch.ones(bs, values.shape[1], dtype=torch.bool, device=x.device)
        abq_pairs, expanded_values, expanded_mask = self.group.lift((coords, values, mask), self.liftsamples)
        _, out_values, out_mask = super().forward((abq_pairs, expanded_values, expanded_mask))
        out_values = out_values.reshape(bs, h * w, self.liftsamples, -1)
        out_mask = out_mask.reshape(bs, h * w, self.liftsamples)
        out_mask = out_mask.unsqueeze(-1)
        summed = (out_values * out_mask).sum(dim=2)
        denom = out_mask.sum(dim=2).clamp(min=1)
        pooled = summed / denom
        return pooled.reshape(bs, h, w, -1).permute(0, 3, 1, 2)


class FisheyeConv(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
        camera: Optional[FisheyeCameraModel] = None,
        group: Optional[FisheyeSO3] = None,
        liftsamples=4,
        mc_samples=32,
        fill=1 / 4,
        knn=True,
    ):
        super().__init__()
        del k, p, g, d
        self.conv = FisheyeLieConv2d(
            c1,
            c2,
            stride=s,
            camera=camera,
            group=group,
            liftsamples=liftsamples,
            mc_samples=mc_samples,
            fill=fill,
            knn=knn,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
