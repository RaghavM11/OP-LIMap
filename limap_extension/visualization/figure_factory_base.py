from abc import abstractmethod
from typing import TYPE_CHECKING, List, Dict, Optional, Union

import torch
import numpy as np
import open3d as o3d

# from arm_clouds import PointCloudList
from limap_extension.visualization.plotly.mesh_viz import MeshViz
from limap_extension.visualization.plotly.gripper_mesh_viz import GripperMeshViz
from limap_extension.visualization.plotly.util import FRAME_DURATION_MS_DEFAULT

# if TYPE_CHECKING:
#     from arm_clouds import PointCloud
#     from cdcpd_torch.data_utils.types.grippers import GrippersInfo


class FigureFactoryBase:
    """Base class factory for producing single- and multi-timestep figures"""

    def __init__(self, frame_duration_ms: float = FRAME_DURATION_MS_DEFAULT):
        # Duration of each frame in milliseconds. To my knowledge, plotly doesn't support setting a
        # per-frame duration so this won't be totally accurate for real world data. Will have to use
        # RVIZ or something similar to get totally accurate visualization.
        self._frame_duration_ms = frame_duration_ms

        # The number of frames in the visualization. Keeps track of this so as to check that newly
        # added things to visualize have the same number of frames.
        self._num_frames: int = -1

        # Store the scene bounds as mins/maxes so that when making the figure we can find the axis
        # with the highest range and make a scene with a cube aspect ratio.
        self._scene_bounds_min: np.ndarray = np.full(3, np.finfo(np.float64).max)
        self._scene_bounds_max: np.ndarray = np.full(3, np.finfo(np.float64).min)

        # TODO: Switch to holding a list of `DeformableObjectTracking` objects so that we can
        # visualize cloth.
        self._tracking_history: Dict[str, List['PointCloudList']] = dict()
        self._tracking_edges: Dict[str, Union[torch.Tensor, np.ndarray]] = dict()

        # The point clouds to visualize.
        self._clouds: Dict[str, 'PointCloudList'] = dict()

        # Meshes that don't change frame-to-frame that we'll visualize. We store these in a list
        # since the `MeshViz` class stores the name that we'll visualize.
        self._mesh_vizs_static: List[MeshViz] = []

        # Meshes that change from frame-to-frame.
        self._mesh_vizs_dynamic: Dict[str, List[MeshViz]] = dict()

    @abstractmethod
    def make_fig(self, is_verbose: bool = False):
        pass

    def set_frame_duration_ms(self, fdms: float):
        if fdms < 10:
            print(f"Warning: setting frame duration to {fdms} milliseconds which is fast. This "
                  f"corresponds to {1000. / fdms} frames per second.")
        self._frame_duration_ms = fdms

    def set_frame_duration_secs(self, fds: float):
        self.set_frame_duration_ms(fds * 1e3)

    def get_num_obstacle_meshes(self) -> int:
        return len(self._mesh_vizs_static)

    def get_visualized_item_names(self) -> Dict[str, List[str]]:
        names = dict()
        names["Tracking History"] = list(self._tracking_history.keys())
        names["Point Clouds"] = list(self._clouds.keys())
        names["Static Meshes"] = [m.name for m in self._mesh_vizs_static]
        names["Dynamic Meshes"] = list(self._mesh_vizs_dynamic.keys())

        for key in list(names.keys()):
            if len(names[key]) == 0:
                names.pop(key)

        return names

    def add_tracking_history(self,
                             name: str,
                             verts_history: 'PointCloudList',
                             edges: Optional[Union[torch.Tensor, np.ndarray]] = None):
        """Adds tracking history to visualization

        TODO: This assumes rope as of right now but we should change the visualization function to
        expect `DeformableObjectTracking` objects and use those to form the keyword arguments for
        making plotly visualizations.
        """
        self._tracking_history[name] = verts_history

        if isinstance(edges, torch.Tensor):
            edges = edges.detach().numpy()
        self._tracking_edges[name] = edges

        self._check_frame_length(name, len(verts_history))

        bounds = verts_history.get_xyz_bounds_numpy()
        self._update_bounds(*bounds)

    def add_cloud(self, name: str, cloud_list: 'PointCloudList'):
        """Adds a list of point clouds to the visualization"""
        self._clouds[name] = cloud_list

        self._check_frame_length(name, len(cloud_list))

        bounds = cloud_list.get_xyz_bounds_numpy()
        self._update_bounds(*bounds)

    def add_obstacle(self, name: str, mesh: o3d.geometry.TriangleMesh):
        """Adds an obstacle mesh to the visualization."""
        mv = MeshViz(name, mesh)
        self._add_mesh_static(mv)

    def add_obstacles(self, mesh_list: List[o3d.geometry.TriangleMesh]):
        num_obstacles_init = self.get_num_obstacle_meshes()
        for i, m in mesh_list:
            name = f"Obstacle #{num_obstacles_init + i}"
            self.add_obstacle(name, m)

    # def add_grippers(self,
    #                  grippers_info_list: List['GrippersInfo'],
    #                  gripper_mesh: Optional[o3d.geometry.TriangleMesh] = None,
    #                  add_axis_to_mesh: bool = False):
    #     """Add grippers to visualize

    #     Makes the assumption that a gripper is present from the start to end. This could be extended
    #     to visualize grippers that pip in/out of existence, but I don't see a big use case for that.
    #     """
    #     gripper_meshes = dict()

    #     for i, grippers_info in enumerate(grippers_info_list):
    #         for j, gripper in enumerate(grippers_info):
    #             name = f"Gripper #{j}"
    #             if i == 0:
    #                 gripper_meshes[name] = []

    #             gripper_meshes[name].append(
    #                 GripperMeshViz(name, gripper, gripper_mesh, add_axis_to_mesh))

    #     for name, mesh_list in gripper_meshes.items():
    #         self._add_mesh_dynamic(name, mesh_list)

    def print_visualized_items(self):
        print(f"Constructing visualization of {self._num_frames} frames of the following items:")
        for category_name, item_names in self.get_visualized_item_names().items():
            print(f"\t{category_name}")
            for n in item_names:
                print(f"\t\t{n}")

    # def _add_mesh_static(self, mesh: MeshViz):
    #     """Adds a mesh that doesn't change pose throughout the visualization"""
    #     if mesh.name in [m.name for m in self._mesh_vizs_static]:
    #         raise KeyError(f"Already added data corresponding to mesh with name, '{mesh.name}'")

    #     self._update_bounds_from_mesh(mesh)
    #     self._mesh_vizs_static.append(mesh)

    # def _add_mesh_dynamic(self, name: str, mvl: List[MeshViz]):
    #     """Adds a list of meshes corresponding to a single mesh that changes throughout time"""
    #     if name in self._mesh_vizs_dynamic.keys():
    #         raise KeyError(f"Already added data corresponding to mesh list with name, '{name}'.")

    #     self._check_frame_length(name, len(mvl))

    #     # If this becomes too computationally intensive, I could always only check every 2, 3, etc.
    #     # mesh in the list which seems reasonable given an assumption of small motion between
    #     # frames.
    #     for m in mvl:
    #         self._update_bounds_from_mesh(m)

    #     self._mesh_vizs_dynamic[name] = mvl

    # def _update_bounds_from_mesh(self, m: 'MeshViz'):
    #     """Updates the bounds of the visualization given a mesh"""
    #     self._update_bounds(m.get_min_bounds(), m.get_max_bounds())

    def _update_bounds(self, xyz_mins: np.ndarray, xyz_maxes: np.ndarray):
        """Updates the bounds of the visualization given an item's min/max coordinates"""
        min_stacked = np.stack((self._scene_bounds_min, xyz_mins), axis=0)
        self._scene_bounds_min = np.min(min_stacked, axis=0)

        max_stacked = np.stack((self._scene_bounds_max, xyz_maxes), axis=0)
        self._scene_bounds_max = np.max(max_stacked, axis=0)

    def _check_frame_length(self, name: str, num_frames_new: int):
        """Verifies that the item to be visualized has the correct number of frames"""
        if self._num_frames == -1:
            self._num_frames = num_frames_new

        if num_frames_new < self._num_frames:
            print(f"Warning: '{name}' had less frames than expected. "
                  f"Had {num_frames_new} frames compared to expected {self._num_frames}. "
                  "This will cut visualization data short for other visualized items.")
            self._num_frames = num_frames_new
        elif num_frames_new > self._num_frames:
            print(f"Warning: '{name}' had more frames than expected. "
                  f"Had {num_frames_new} frames compared to expected {self._num_frames}. "
                  "Won't be able to visualize all items.")
