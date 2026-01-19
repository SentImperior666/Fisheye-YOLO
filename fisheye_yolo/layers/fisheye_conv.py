from typing import Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fisheye_yolo.geometry.fisheye_camera import FisheyeCameraModel
from fisheye_yolo.utils.third_party import ensure_lieconv_on_path

ensure_lieconv_on_path()

from lie_conv.lieConv import LieConv
from lie_conv.lieGroups import FisheyeSO3, norm


def _pixel_grid(h, w, device, dtype):
    """Create a grid of pixel coordinates (u, v) for an h x w image."""
    u = torch.arange(w, device=device, dtype=dtype)
    v = torch.arange(h, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([uu, vv], dim=-1).reshape(-1, 2)


def _grid_neighbor_offsets(radius: int, device, dtype) -> torch.Tensor:
    """
    Precompute relative offsets for a circular neighborhood of given radius.
    Returns tensor of shape (k, 2) where k is the number of neighbors.
    """
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dx, dy))
    return torch.tensor(offsets, device=device, dtype=dtype)


def _build_neighbor_indices(h: int, w: int, radius: int, device) -> torch.Tensor:
    """
    Build neighbor index tensor for a grid of size h x w.
    
    Returns:
        neighbor_idx: (h*w, k) tensor of neighbor indices, -1 for invalid neighbors
    """
    offsets = _grid_neighbor_offsets(radius, device, torch.long)
    k = offsets.shape[0]
    n = h * w
    
    # Create base grid indices
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    yy = yy.reshape(-1, 1)  # (n, 1)
    xx = xx.reshape(-1, 1)  # (n, 1)
    
    # Add offsets to get neighbor coordinates
    ny = yy + offsets[:, 1].unsqueeze(0)  # (n, k)
    nx = xx + offsets[:, 0].unsqueeze(0)  # (n, k)
    
    # Check bounds
    valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    
    # Convert to linear indices
    neighbor_idx = ny * w + nx
    neighbor_idx = torch.where(valid, neighbor_idx, torch.full_like(neighbor_idx, -1))
    
    return neighbor_idx


class FisheyeLieConv2d(LieConv):
    """
    Fisheye-equivariant convolution layer using LieConv with SO(3) group.
    
    Supports multiple processing modes:
    - "patch": Process image in overlapping patches (default, memory efficient)
    - "sparse": Use sparse local neighborhoods (experimental)
    - "full": Original full pairwise computation (only for tiny images)
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 1,
        camera: Optional[FisheyeCameraModel] = None,
        group: Optional[FisheyeSO3] = None,
        liftsamples: int = 4,
        mc_samples: int = 32,
        fill: float = 1 / 4,
        knn: bool = True,
        mode: Literal["patch", "sparse", "full"] = "patch",
        patch_size: int = 32,
        patch_overlap: float = 0.0,
        neighborhood_radius: int = 5,
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
        self.c_out = c2
        
        # Scalability parameters
        self.mode = mode
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.neighborhood_radius = neighborhood_radius
        
        # Cache for neighbor indices (sparse mode)
        self._cached_neighbor_idx = None
        self._cached_grid_size = None

    def _forward_init(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple forward for model initialization only.
        
        Ultralytics runs a test forward pass with tiny dummy inputs (e.g., 64x64)
        to compute layer strides. We use a simple conv for this since LieConv
        is expensive and unnecessary for stride computation.
        """
        bs, c, h, w = x.shape
        if not hasattr(self, '_init_conv') or self._init_conv is None:
            self._init_conv = nn.Conv2d(c, self.c_out, kernel_size=3, padding=1, bias=False).to(x.device).to(x.dtype)
        return self._init_conv(x)

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Original full pairwise computation. Only suitable for very small images.
        Warning: O(n^2) memory where n = h*w*liftsamples.
        """
        bs, c, h, w = x.shape
        coords = _pixel_grid(h, w, x.device, x.dtype).unsqueeze(0).expand(bs, -1, -1)
        values = x.permute(0, 2, 3, 1).reshape(bs, -1, c)
        mask = torch.ones(bs, values.shape[1], dtype=torch.bool, device=x.device)
        
        abq_pairs, expanded_values, expanded_mask = self.group.lift(
            (coords, values, mask), self.liftsamples
        )
        _, out_values, out_mask = super().forward((abq_pairs, expanded_values, expanded_mask))
        
        out_values = out_values.reshape(bs, h * w, self.liftsamples, -1)
        out_mask = out_mask.reshape(bs, h * w, self.liftsamples).unsqueeze(-1)
        
        summed = (out_values * out_mask).sum(dim=2)
        denom = out_mask.sum(dim=2).clamp(min=1)
        pooled = summed / denom
        
        return pooled.reshape(bs, h, w, -1).permute(0, 3, 1, 2)

    def _forward_patch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image in overlapping patches with mini-batching.
        
        Processes patches in small groups (patch_batch_size) to balance
        GPU utilization with memory constraints.
        """
        bs, c, h, w = x.shape
        
        # Adjust patch size if feature map is smaller than configured patch_size
        # This happens in deeper layers where spatial resolution is reduced
        P = min(self.patch_size, h, w)
        
        # If feature map is small enough, process directly without patching
        if h <= P and w <= P:
            return self._forward_full(x)
        
        S = P  # Non-overlapping for speed
        
        # Pad image if needed to fit patches evenly
        pad_h = (P - h % P) % P
        pad_w = (P - w % P) % P
        if pad_h > 0 or pad_w > 0:
            # Use 'constant' padding if reflect would fail (pad >= dim)
            pad_mode = 'reflect' if (pad_h < h and pad_w < w) else 'constant'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        
        h_padded, w_padded = x.shape[2], x.shape[3]
        n_patches_h = h_padded // P
        n_patches_w = w_padded // P
        
        # Initialize output
        output = torch.zeros(bs, self.c_out, h_padded, w_padded, device=x.device, dtype=x.dtype)
        
        # Process patches in small batches (4 at a time for memory efficiency)
        patch_batch_size = 4
        
        patch_coords = [(i, j) for i in range(n_patches_h) for j in range(n_patches_w)]
        
        for batch_start in range(0, len(patch_coords), patch_batch_size):
            batch_end = min(batch_start + patch_batch_size, len(patch_coords))
            batch_coords = patch_coords[batch_start:batch_end]
            
            # Collect patches for this mini-batch
            patches_list = []
            for (i, j) in batch_coords:
                y_start, x_start = i * P, j * P
                patch = x[:, :, y_start:y_start+P, x_start:x_start+P]  # (bs, c, P, P)
                patches_list.append(patch)
            
            # Stack into (num_patches_in_batch, bs, c, P, P) then reshape
            patches_stack = torch.stack(patches_list, dim=0)  # (npb, bs, c, P, P)
            npb = patches_stack.shape[0]
            patches_batch = patches_stack.reshape(npb * bs, c, P, P)  # (npb*bs, c, P, P)
            
            # Process through LieConv
            out_batch = self._forward_full(patches_batch)  # (npb*bs, c_out, P, P)
            out_batch = out_batch.reshape(npb, bs, self.c_out, P, P)  # (npb, bs, c_out, P, P)
            
            # Place outputs back
            for idx, (i, j) in enumerate(batch_coords):
                y_start, x_start = i * P, j * P
                output[:, :, y_start:y_start+P, x_start:x_start+P] = out_batch[idx]
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]
        
        return output

    def _get_neighbor_indices(self, h: int, w: int, device) -> torch.Tensor:
        """Get or compute cached neighbor indices for sparse mode."""
        if self._cached_neighbor_idx is None or self._cached_grid_size != (h, w):
            self._cached_neighbor_idx = _build_neighbor_indices(
                h, w, self.neighborhood_radius, device
            )
            self._cached_grid_size = (h, w)
        return self._cached_neighbor_idx

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process with sparse local neighborhoods only.
        Memory efficient: O(n * k * liftsamples) where k = neighborhood size.
        """
        bs, c, h, w = x.shape
        n = h * w
        device, dtype = x.device, x.dtype
        
        # Create pixel coordinates and values
        coords = _pixel_grid(h, w, device, dtype)  # (n, 2)
        values = x.permute(0, 2, 3, 1).reshape(bs, n, c)  # (bs, n, c)
        
        # Unproject to rays and compute lifted elements
        rays = self.group.camera.unproject(coords)  # (n, 3)
        
        # Compute lifted elements for each point (without full pairwise)
        lifted_a, _ = self.group.lifted_elems(
            rays.unsqueeze(0).expand(bs, -1, -1),
            self.liftsamples
        )  # (bs, n*liftsamples, 3)
        
        n_lifted = n * self.liftsamples
        
        # Expand values to match liftsamples
        expanded_values = values.unsqueeze(2).expand(-1, -1, self.liftsamples, -1)
        expanded_values = expanded_values.reshape(bs, n_lifted, c)
        
        # Get neighbor indices for the grid
        neighbor_idx = self._get_neighbor_indices(h, w, device)  # (n, k)
        k = neighbor_idx.shape[1]
        
        # Expand neighbor indices for liftsamples
        # Each lifted point at pixel i should look at lifted points at neighbor pixels
        # neighbor_idx: (n, k) -> (n * liftsamples, k * liftsamples)
        neighbor_idx_lifted = neighbor_idx.unsqueeze(1).expand(-1, self.liftsamples, -1)
        neighbor_idx_lifted = neighbor_idx_lifted * self.liftsamples  # base index
        
        # Add offsets for each liftsample
        lift_offsets = torch.arange(self.liftsamples, device=device)
        neighbor_idx_lifted = neighbor_idx_lifted.unsqueeze(-1) + lift_offsets
        neighbor_idx_lifted = neighbor_idx_lifted.reshape(n * self.liftsamples, k * self.liftsamples)
        
        # Mark invalid neighbors
        invalid_mask = neighbor_idx.unsqueeze(1).expand(-1, self.liftsamples, -1) < 0
        invalid_mask = invalid_mask.unsqueeze(-1).expand(-1, -1, -1, self.liftsamples)
        invalid_mask = invalid_mask.reshape(n_lifted, k * self.liftsamples)
        
        # Gather neighbor lifted elements
        # lifted_a: (bs, n_lifted, 3)
        B = torch.arange(bs, device=device)[:, None, None]
        neighbor_idx_safe = neighbor_idx_lifted.clamp(min=0).unsqueeze(0).expand(bs, -1, -1)
        
        neighbor_lifted_a = lifted_a[B, neighbor_idx_safe]  # (bs, n_lifted, k*liftsamples, 3)
        neighbor_values = expanded_values[B, neighbor_idx_safe]  # (bs, n_lifted, k*liftsamples, c)
        
        # Compute pairwise Lie algebra differences: log(exp(-b) @ exp(a))
        a = lifted_a.unsqueeze(2)  # (bs, n_lifted, 1, 3)
        b = neighbor_lifted_a      # (bs, n_lifted, k*liftsamples, 3)
        
        # Use group exp/log for proper SO(3) difference
        exp_neg_b = self.group.exp(-b)  # (bs, n_lifted, k*liftsamples, 3, 3)
        exp_a = self.group.exp(a)        # (bs, n_lifted, 1, 3, 3)
        ab_pairs = self.group.log(exp_neg_b @ exp_a)  # (bs, n_lifted, k*liftsamples, 3)
        
        # Create neighbor mask
        neighbor_mask = ~invalid_mask.unsqueeze(0).expand(bs, -1, -1)  # (bs, n_lifted, k*liftsamples)
        
        # Apply the weightnet and convolution
        # This mimics LieConv.point_convolve but with our sparse structure
        _, penult_weights, _ = self.weightnet((None, ab_pairs, neighbor_mask))
        
        # Mask invalid neighbors
        penult_weights = torch.where(
            neighbor_mask.unsqueeze(-1),
            penult_weights,
            torch.zeros_like(penult_weights)
        )
        neighbor_values_masked = torch.where(
            neighbor_mask.unsqueeze(-1),
            neighbor_values,
            torch.zeros_like(neighbor_values)
        )
        
        # Compute convolution: (bs, n_lifted, c, k*ls) @ (bs, n_lifted, k*ls, cmco/c) -> (bs, n_lifted, cmco)
        partial_conv = (neighbor_values_masked.transpose(-1, -2) @ penult_weights)
        partial_conv = partial_conv.reshape(bs, n_lifted, -1)
        conv_out = self.linear(partial_conv)  # (bs, n_lifted, c_out)
        
        # Pool over liftsamples
        conv_out = conv_out.reshape(bs, n, self.liftsamples, -1)
        pooled = conv_out.mean(dim=2)  # (bs, n, c_out)
        
        # Reshape to image
        return pooled.reshape(bs, h, w, -1).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using LieConv for fisheye-equivariant convolution.
        
        Args:
            x: Input tensor of shape (bs, c, h, w)
            
        Returns:
            Output tensor of shape (bs, c_out, h, w)
        
        Modes:
            - "patch": Process in patches (default, memory efficient)
            - "sparse": Sparse neighborhoods (experimental)
            - "full": Full pairwise LieConv (small images only)
        """
        # Apply stride via pooling first
        if self.stride > 1:
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        
        bs, c, h, w = x.shape
        
        # Use simple conv for model initialization only (stride computation)
        # Ultralytics uses bs=1, not training, with small dummy images
        is_init_pass = (bs == 1 and not self.training and h <= 64 and w <= 64)
        if is_init_pass:
            return self._forward_init(x)
        
        # Dispatch to LieConv implementation
        if self.mode == "patch":
            return self._forward_patch(x)
        elif self.mode == "sparse":
            return self._forward_sparse(x)
        elif self.mode == "full":
            return self._forward_full(x)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'patch', 'sparse', or 'full'.")


class FisheyeConv(nn.Module):
    """
    Fisheye-equivariant convolution block matching YOLO Conv interface.
    Includes BatchNorm and activation.
    
    When camera is not provided, uses the global camera from
    fisheye_yolo.utils.global_camera.get_global_fisheye_camera().
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
        camera: Optional[FisheyeCameraModel] = None,
        group: Optional[FisheyeSO3] = None,
        liftsamples: Optional[int] = None,
        mc_samples: Optional[int] = None,
        fill: Optional[float] = None,
        knn: Optional[bool] = None,
        # Scalability parameters
        mode: Optional[Literal["patch", "sparse", "full"]] = None,
        patch_size: Optional[int] = None,
        patch_overlap: Optional[float] = None,
        neighborhood_radius: Optional[int] = None,
    ):
        super().__init__()
        # These params are not used in LieConv but kept for YOLO compatibility
        del k, p, g, d
        
        # If no camera provided, use global camera
        if camera is None and group is None:
            from fisheye_yolo.utils.global_camera import get_global_fisheye_camera, get_global_fisheye_config
            camera = get_global_fisheye_camera()
            if camera is None:
                raise ValueError(
                    "No camera provided and no global camera set. "
                    "Call fisheye_yolo.utils.global_camera.set_global_fisheye_camera() first."
                )
            # Use global config for unset parameters
            cfg = get_global_fisheye_config()
            if liftsamples is None:
                liftsamples = cfg["liftsamples"]
            if mc_samples is None:
                mc_samples = cfg["mc_samples"]
            if fill is None:
                fill = cfg["fill"]
            if knn is None:
                knn = cfg["knn"]
            if mode is None:
                mode = cfg["mode"]
            if patch_size is None:
                patch_size = cfg["patch_size"]
            if patch_overlap is None:
                patch_overlap = cfg["patch_overlap"]
            if neighborhood_radius is None:
                neighborhood_radius = cfg["neighborhood_radius"]
        else:
            # Use defaults if not specified
            liftsamples = liftsamples if liftsamples is not None else 4
            mc_samples = mc_samples if mc_samples is not None else 32
            fill = fill if fill is not None else 0.25
            knn = knn if knn is not None else True
            mode = mode if mode is not None else "patch"
            patch_size = patch_size if patch_size is not None else 16
            patch_overlap = patch_overlap if patch_overlap is not None else 0.5
            neighborhood_radius = neighborhood_radius if neighborhood_radius is not None else 5
        
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
            mode=mode,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            neighborhood_radius=neighborhood_radius,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
