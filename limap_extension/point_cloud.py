from typing import Union, Optional

import torch
import numpy as np

try:
    import open3d as o3d
    OPEN3D_FOUND = True
except ImportError:
    OPEN3D_FOUND = False

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseTypeMixin(ABC):
    """The base type mix-in that all CDCPD-types inherit from.

    NOTE: All type containers should inherit from either UserList or UserDict *and* BaseTypeMixin.
    This ensures that all containers for types behave as lists or dicts and have the methods
    required for interacting with them defined in this class.
    """
    is_torch: bool

    @abstractmethod
    def to_torch(self, dtype: torch.dtype = torch.double, device: torch.device = "cpu"):
        pass

    @abstractmethod
    def to_numpy(self):
        pass

    @abstractmethod
    def copy(self):
        pass


class PointCloud(BaseTypeMixin):
    DEFAULT_PLOTLY_POINT_CLOUD_COLOR: str = 'yellow'
    """Simple class for storage of point clouds"""

    def __init__(self,
                 xyz: Union[torch.Tensor, np.ndarray],
                 rgb: Optional[Union[torch.Tensor, np.ndarray]] = None):
        """Initializes the point cloud with XYZ and optional RGB data

        Parameters
        ----------
        xyz : Union[torch.Tensor, np.ndarray]
            The XYZ coordinates of the point cloud. Should be of shape [3, N] where N is the number
            of points.
        rgb : Union[torch.Tensor, np.ndarray], optional
            The RGB values of each point in the cloud. Should be of shape [3, N] if specified. Can
            be normalized (0-1) or unnormalized (0-255).
        """
        # Ensure the point cloud is the correct shape.
        if xyz.shape[1] != 3:
            # Note, this can be ambiguous for an edge case where the point cloud has 3 points in it,
            # thus the shape being 3x3. However, that's rare enough to where the user will get this
            # warning when inputting XYZ clouds with the wrong dimensions eventually.
            raise ValueError(f"Expected `xyz` of shape [3, N] but got shape {xyz.shape} instead")

        self.xyz: Union[torch.Tensor, np.ndarray] = xyz
        self.is_torch: bool = isinstance(self.xyz, torch.Tensor)

        self.rgb: Optional[Union[torch.Tensor, np.ndarray]] = None
        self.has_rgb: bool = False
        self.is_rgb_normalized: bool = False
        self.add_rgb(rgb)

    def to_numpy(self):
        """Converts cloud to Numpy arrays if they're tensors"""
        if self.is_torch:
            self.xyz = self.xyz.detach().numpy()
            if self.has_rgb:
                self.rgb = self.rgb.detach().numpy()
            self.is_torch = False

    def to_torch(self, xyz_dtype: torch.dtype = torch.double, device: torch.device = "cpu"):
        """Converts cloud to tensors if they're numpy array"""
        if not self.is_torch:
            self.xyz = torch.from_numpy(self.xyz).to(xyz_dtype).to(device)
            if self.has_rgb:
                self.rgb = torch.from_numpy(self.rgb).to(device)
            self.is_torch = True
        else:
            # This should be a no-op if the cloud is already a tensor with correct dtype and device.
            self.xyz = self.xyz.to(dtype=xyz_dtype, device=device)
            if self.has_rgb:
                self.rgb = self.rgb.to(device)

    def normalize_rgb(self):
        if self.has_rgb and not self.is_rgb_normalized:
            if self.is_torch:
                rgb = self.rgb.to(dtype=self.xyz.dtype, device=self.xyz.device)
            else:
                rgb = self.rgb.astype(float)
            self.rgb = rgb / 255.
            self.is_rgb_normalized = True

    def unnormalize_rgb(self):
        if self.has_rgb and self.is_rgb_normalized:
            self.rgb *= 255.
            if self.is_torch:
                self.rgb = self.rgb.to(dtype=torch.uint8)
            else:
                self.rgb = self.rgb.astype(np.uint8)
            self.is_rgb_normalized = False

    def add_rgb(self, rgb: Union[torch.Tensor, np.ndarray]):
        if self.has_rgb:
            raise AttributeError("PointCloud already has RGB values!")

        self.rgb = rgb
        self.has_rgb = rgb is not None

        if self.has_rgb:
            assert (type(self.xyz) == type(
                self.rgb)), ("XYZ and RGB clouds must be same array "
                             f"type! XYZ is {type(self.xyz)} and RGB is {type(self.rgb)}")

            rgb_max_val = self.rgb.max() if self.is_torch else np.max(self.rgb)

            self.is_rgb_normalized = rgb_max_val <= 1.0

    def set_uniform_rgb(self, rgb: Union[torch.Tensor, np.ndarray]):
        if rgb.shape[1] != 3 and len(rgb.shape) != 1:
            raise ValueError("Expected RGB numpy array or tensor with shape (3) but got shape:",
                             rgb.shape)

        ones_func = torch.ones_like if self.is_torch else np.ones_like

        rgb_expanded = ones_func(self.xyz) * rgb[None, :]
        self.add_rgb(rgb_expanded)

    # def get_xyz_bounds_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Returns tuple of one dimensional pytorch tensors for axis-wise min/max of coordinates"""
    #     if self.is_torch:
    #         bounds = (self.xyz.min(dim=1).values, self.xyz.max(dim=1).values)
    #     else:
    #         bounds_np = self.get_xyz_bounds_numpy()
    #         bounds = (torch.from_numpy(bounds_np[0]), torch.from_numpy(bounds_np[1]))
    #     return bounds

    # def get_xyz_bounds_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """Returns tuple of one dimensional numpy arrays for axis-wise min/max of coordinates"""
    #     if self.is_torch:
    #         bounds_torch = self.get_xyz_bounds_torch()
    #         bounds = (bounds_torch[0].detach().cpu().numpy(),
    #                   bounds_torch[1].detach().cpu().numpy())
    #     else:
    #         bounds = (np.min(self.xyz, axis=1), np.max(self.xyz, axis=1))
    #     return bounds

    def downsample(self, voxel_size=0.02):
        """Performs voxel grid downsampling"""
        if not OPEN3D_FOUND:
            raise ImportError("Open3D not found. Please install Open3D to use this method.")

        is_torch_orig = self.is_torch
        if is_torch_orig:
            dtype_orig = self.xyz.dtype
            device_orig = self.xyz.device

        is_rgb_normalized_orig = self.is_rgb_normalized

        # Prepare the point cloud for conversion to open3d cloud
        self.to_numpy()
        self.normalize_rgb()

        # Downsample using open3d.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz.T)

        if self.has_rgb:
            pcd.colors = o3d.utility.Vector3dVector(self.rgb.T)

        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        self.xyz = np.asarray(pcd_downsampled.points).T
        self.rgb = np.asarray(pcd_downsampled.colors).T

        # Convert back to original datatype.
        if is_torch_orig:
            self.to_torch(dtype_orig, device_orig)

        if self.has_rgb and (not is_rgb_normalized_orig):
            self.unnormalize_rgb()

    def apply_extrinsic(self, ext: Union[np.ndarray, torch.Tensor]):
        """Applies extrinsic transformation to the point cloud"""
        if ext.shape != (4, 4):
            raise ValueError(
                f"Expected extrinsic matrix of shape (4, 4) but got {ext.shape} instead")

        if self.is_torch:
            if isinstance(ext, np.ndarray):
                ext = torch.from_numpy(ext).to(dtype=self.xyz.dtype, device=self.xyz.device)
        else:
            if isinstance(ext, torch.Tensor):
                ext = ext.detach().numpy()

        self.xyz = (ext[:3, :3] @ self.xyz.T).T + ext[:3, 3].reshape(1, 3)

    def to_open3d_copy(self):
        if not OPEN3D_FOUND:
            raise ImportError("Open3D not found. Please install Open3D to use this method.")

        # Convert copies to numpy arrays if tensors
        xyz = self.xyz
        rgb = self.rgb
        if self.is_torch:
            xyz = xyz.detach().clone()
            if self.has_rgb:
                rgb = self.rgb.detach().clone()

        # Normalize the copies RGB values if it's not already normalized.
        if self.has_rgb and not self.is_rgb_normalized:
            rgb = self.rgb.astype(float) / 255.

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.T)
        if self.has_rgb:
            pcd.colors = o3d.utility.Vector3dVector(rgb.T)

        return pcd

    def visualize_open3d(self):
        pcd = self.to_open3d_copy()
        o3d.visualization.draw_geometries([pcd])

    def form_rerun_kwargs(self, radii: float = 0.01, color: Optional[np.ndarray] = None):
        pts = self.xyz
        rgb = self.rgb

        # Determine color.
        if color is not None:
            rgb = color
        elif self.has_rgb:
            if self.is_torch:
                rgb = rgb.detach().numpy()
            if not self.is_rgb_normalized:
                rgb = rgb.astype(float) / 255.
        else:
            num_pts = pts.shape[0]
            rgb = np.array((1.0, 0.0, 0.0)).reshape(1, 3).repeat(num_pts, 1)

        # Prep the points for logging.
        if self.is_torch:
            pts = pts.detach().numpy()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().numpy()

        kwargs = {"positions": pts, "colors": rgb, "radii": radii}

        return kwargs

    def copy(self) -> 'PointCloud':
        if self.is_torch:
            xyz_new = self.xyz.detach().clone()
            if self.has_rgb:
                rgb_new = self.rgb.detach().clone()
            else:
                rgb_new = None
        else:
            xyz_new = self.xyz.copy()
            if self.has_rgb:
                rgb_new = self.rgb.copy()
            else:
                rgb_new = None

        pc_new = PointCloud(xyz_new, rgb_new)
        return pc_new
