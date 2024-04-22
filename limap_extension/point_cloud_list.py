from typing import Union, List, Tuple, Iterable
from warnings import warn

import torch
import numpy as np
import plotly.graph_objects as go

from .base_type_list_mixin import BaseTypeList
from .point_cloud import PointCloud
# from .plotly_util import (make_update_menus, make_sliders, FRAME_DURATION_MS_DEFAULT)


class PointCloudList(BaseTypeList[PointCloud]):
    """Container for `PointCloud` storage"""

    def __init__(self, initlist=None):
        # Special constructors just for PointCloudList
        # NOTE: This assumes just XYZ data. For XYZ and RGB, use the append_create_cloud() method.
        if isinstance(initlist, (torch.Tensor, np.ndarray)):
            super().__init__()
            self._from_xyz_matrix(initlist)
        elif (isinstance(initlist, Iterable)
              and all([isinstance(it, (torch.Tensor, np.ndarray)) for it in initlist])):
            super().__init__()
            for xyz in initlist:
                self.append_create_cloud(xyz)
        else:
            super().__init__(initlist)

    def _get_item_type(self):
        return PointCloud

    def _check_item_attributes(self, item: PointCloud):
        """Verifies that the item has RGB if the list does, and vice-versa"""
        if not self._is_initialized():
            return True

        return (len(self) == 0) or (self.has_rgb() == item.has_rgb)

    def _from_xyz_matrix(self, xyzs: Union[torch.Tensor, np.ndarray]):
        if len(xyzs.shape) != 3 or xyzs.shape[1] != 3:
            raise ValueError("Expected shape of [num_clouds, 3, num_points]")
        num_timesteps = xyzs.shape[0]
        for i in range(num_timesteps):
            pts = xyzs[i, ...]
            self.append_create_cloud(pts)

    @staticmethod
    def from_xyz_matrix(xyzs: Union[torch.Tensor, np.ndarray]):
        """Returns PointCloudList from matrix with shape [num_timesteps, 3, num_pts]"""
        warn("PointCloudList.from_xyz_matrix is deprecated. Use constructor (__init__) instead.")
        return PointCloudList(xyzs)

    @staticmethod
    def from_cloud_list(xyzs_list: List[Union[torch.Tensor, np.ndarray]]):
        warn("PointCloudList.from_cloud_list is deprecated. Use constructor (__init__) instead.")
        return PointCloudList(xyzs_list)

    def get_xyz_cloud_list(self) -> List[Union[torch.Tensor, np.ndarray]]:
        """Returns an actual list of just xyz values"""
        return [c.xyz for c in self]

    def get_xyz_bounds_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns tuple of one dimensional pytorch tensors for axis-wise min/max of coordinates"""
        if not self.is_torch():
            bounds_np = self.get_xyz_bounds_numpy()
            bounds = (torch.from_numpy(bounds_np[0]), torch.from_numpy(bounds_np[1]))
        else:
            bounds_list = [c.get_xyz_bounds_torch() for c in self]

            mins_list = [b[0] for b in bounds_list]
            min_bounds = torch.stack(mins_list, dim=0)

            maxes_list = [b[1] for b in bounds_list]
            max_bounds = torch.stack(maxes_list, dim=0)

            bounds = (min_bounds.min(dim=0).values, max_bounds.max(dim=0).values)

        return bounds

    def get_xyz_bounds_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns tuple of one dimensional numpy arrays for axis-wise min/max of coordinates"""
        if not self.is_torch():
            bounds_list = [c.get_xyz_bounds_numpy() for c in self]

            mins_list = [b[0] for b in bounds_list]
            min_bounds = np.stack(mins_list, axis=0)

            maxes_list = [b[1] for b in bounds_list]
            max_bounds = np.stack(maxes_list, axis=0)

            bounds = (np.min(min_bounds, axis=0), np.max(max_bounds, axis=0))
        else:
            bounds_torch = self.get_xyz_bounds_torch()
            bounds = (bounds_torch[0].detach().cpu().numpy(),
                      bounds_torch[1].detach().cpu().numpy())

        return bounds

    def has_rgb(self) -> bool:
        """Returns True if all clouds have RGB data or if no clouds stored"""
        return all([c.has_rgb for c in self])

    def append_create_cloud(self,
                            xyz: Union[torch.Tensor, np.ndarray],
                            rgb: Union[torch.tensor, np.ndarray] = None) -> None:
        """Creates and appends `PointCloud`"""
        pc = PointCloud(xyz, rgb)
        self.append(pc)

    def unnormalize_rgb(self):
        """Un-normalizes the RGB portions of the clouds"""
        for c in self:
            c.unnormalize_rgb()

    def normalize_rgb(self):
        """Normalizes the RGB portions of the clouds"""
        for c in self:
            c.normalize_rgb()

    def set_uniform_rgb(self, rgb: Union[np.ndarray, torch.Tensor]):
        """Sets a uniform RGB value for all points in all clouds"""
        if rgb.shape[0] != 3 and len(rgb.shape) != 1:
            raise ValueError("Expected RGB numpy array or tensor with shape (3) but got shape:",
                             rgb.shape)

        for c in self:
            c.set_uniform_rgb(rgb)

    # def visualize_plotly(self, frame_duration_ms: float = FRAME_DURATION_MS_DEFAULT):
    #     # Get a copy so we don't modify the original list, then prep for visualization.
    #     pcl_copy: PointCloudList = self.copy()
    #     pcl_copy.to_numpy()
    #     pcl_copy.normalize_rgb()

    #     num_clouds = len(pcl_copy)

    #     ## Making the layout.
    #     ## ------------------
    #     xyz_mins = np.stack([np.min(c.xyz, axis=1) for c in pcl_copy], axis=0)
    #     xyz_maxes = [np.max(c.xyz, axis=1) for c in pcl_copy]
    #     xyz_min = np.min(xyz_mins, axis=0)
    #     xyz_max = np.max(xyz_maxes, axis=0)
    #     xyz_mid = (xyz_min + xyz_max) / 2.0

    #     half_ranges = (xyz_max - xyz_min) / 2.0
    #     half_range_max = np.max(half_ranges)

    #     ranges_min = xyz_mid - half_range_max
    #     ranges_max = xyz_mid + half_range_max

    #     scene = go.layout.Scene(xaxis=go.layout.scene.XAxis(range=(ranges_min[0], ranges_max[0]),
    #                                                         autorange=False),
    #                             yaxis=go.layout.scene.YAxis(range=(ranges_min[1], ranges_max[1]),
    #                                                         autorange=False),
    #                             zaxis=go.layout.scene.ZAxis(range=(ranges_min[2], ranges_max[2]),
    #                                                         autorange=False),
    #                             aspectmode="cube")

    #     updatemenus = make_update_menus(frame_duration_ms=frame_duration_ms)
    #     sliders = make_sliders(num_clouds, frame_duration_ms=frame_duration_ms)
    #     layout = go.Layout(scene=scene,
    #                        width=750,
    #                        height=750,
    #                        updatemenus=updatemenus,
    #                        sliders=[sliders])

    #     ## Making the animation frames
    #     ## ---------------------------
    #     frames_animation = []
    #     for i, c in enumerate(pcl_copy):
    #         data = []
    #         kwargs_scatter = c.form_plotly_scatter3d_kwargs(trace_name="Point Cloud")
    #         data.append(go.Scatter3d(**kwargs_scatter))

    #         frames_animation.append(go.Frame(data=data, layout=layout, name=i))

    #     ## Making initial plot data
    #     ## ------------------------
    #     fig_data_init = []
    #     c: PointCloud = pcl_copy[0]
    #     kwargs_scatter_init = c.form_plotly_scatter3d_kwargs(trace_name="Point Cloud")
    #     fig_data_init.append(go.Scatter3d(kwargs_scatter_init))

    #     fig = go.Figure(data=fig_data_init, layout=layout, frames=frames_animation)
    #     fig.show()
