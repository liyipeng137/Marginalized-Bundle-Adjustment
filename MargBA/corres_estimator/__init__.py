from .corre_utils import inference_pairwise, two_view_pose_estimation, two_view_adjustment_estimation, torchncoords2coordinates, npose2pose


__all__ = [
    "RoMa",
    "MASt3R",
    "MASt3RFast",
    "inference_pairwise",
    "two_view_pose_estimation",
    "two_view_adjustment_estimation",
    "torchncoords2coordinates",
    "npose2pose",
]


def __getattr__(name):
    if name == "RoMa":
        from .roma import RoMa
        return RoMa
    if name == "MASt3R":
        from .mast3r import MASt3R
        return MASt3R
    if name == "MASt3RFast":
        from .mast3r_fast import MASt3RFast
        return MASt3RFast
    raise AttributeError(f"module {__name__} has no attribute {name}")
