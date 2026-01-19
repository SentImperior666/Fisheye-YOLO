"""
Global camera registry for FisheyeConv layers.

Since YAML configs can't pass complex objects like camera models,
we use a global registry that FisheyeConv layers can access.
"""
from typing import Optional
from fisheye_yolo.geometry import FisheyeCameraModel

# Global camera instance
_GLOBAL_CAMERA: Optional[FisheyeCameraModel] = None

# Global fisheye layer config
_GLOBAL_CONFIG = {
    "liftsamples": 4,
    "mc_samples": 32,
    "fill": 0.25,
    "knn": True,
    "mode": "patch",
    "patch_size": 32,
    "patch_overlap": 0.0,
    "neighborhood_radius": 5,
}


def set_global_fisheye_camera(
    fx: float = 320.0,
    fy: float = 320.0,
    cx: float = 320.0,
    cy: float = 320.0,
    model: str = "equisolid",
    flip_y: bool = True,
    **config_kwargs,
) -> FisheyeCameraModel:
    """
    Set the global camera for FisheyeConv layers.
    
    Args:
        fx, fy: Focal lengths
        cx, cy: Principal point
        model: Fisheye projection model ('equidistant', 'equisolid', 'stereographic', 'orthographic')
        flip_y: Whether to flip Y axis
        **config_kwargs: Additional config for FisheyeConv (liftsamples, mc_samples, etc.)
    
    Returns:
        The created FisheyeCameraModel
    """
    global _GLOBAL_CAMERA, _GLOBAL_CONFIG
    
    _GLOBAL_CAMERA = FisheyeCameraModel(
        fx=fx, fy=fy, cx=cx, cy=cy, model=model, flip_y=flip_y
    )
    
    # Update config with any provided kwargs
    for key, value in config_kwargs.items():
        if key in _GLOBAL_CONFIG:
            _GLOBAL_CONFIG[key] = value
    
    print(f"Global fisheye camera set: {model} (fx={fx}, fy={fy}, cx={cx}, cy={cy})")
    print(f"Global fisheye config: mode={_GLOBAL_CONFIG['mode']}, patch_size={_GLOBAL_CONFIG['patch_size']}")
    
    return _GLOBAL_CAMERA


def get_global_fisheye_camera() -> Optional[FisheyeCameraModel]:
    """Get the global camera for FisheyeConv layers."""
    return _GLOBAL_CAMERA


def get_global_fisheye_config() -> dict:
    """Get the global config for FisheyeConv layers."""
    return _GLOBAL_CONFIG.copy()


def clear_global_fisheye_camera():
    """Clear the global camera (useful for testing)."""
    global _GLOBAL_CAMERA
    _GLOBAL_CAMERA = None
