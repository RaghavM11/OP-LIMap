import numpy as np


class PointCloud:

    def __init__(self, xyz: np.ndarray, rgb: np.ndarray):
        self.xyz: np.ndarray = self._verify_shape(xyz)
        if rgb.dtype != np.uint8:
            raise ValueError("RGB array must be of type uint8")
        self.rgb: np.ndarray = self._verify_shape(rgb)

    def _verify_shape(self, arr: np.ndarray) -> np.ndarray:
        if not arr.ndim == 2:
            raise ValueError("Array must be 2D")
        if not arr.shape[1] == 3:
            raise ValueError(f"Array convention is [N_points, 3], got {arr.shape} instead.")
        return arr
