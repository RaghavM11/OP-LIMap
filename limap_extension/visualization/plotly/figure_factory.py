from typing import TYPE_CHECKING

import plotly.graph_objects as go
import numpy as np

from limap_extension.visualization.figure_factory_base import FigureFactoryBase
from limap_extension.visualization.plotly.util import (make_sliders, make_update_menus,
                                                       FRAME_DURATION_MS_DEFAULT,
                                                       TRACKING_MARKER_SIZE, POINT_CLOUD_OPACITY,
                                                       POINT_CLOUD_MARKER_SIZE)

if TYPE_CHECKING:
    # from arm_clouds import PointCloud
    pass


class FigureFactory(FigureFactoryBase):
    """Factory for producing single- and multi-timestep plotly figures"""

    def __init__(self, frame_duration_ms: float = FRAME_DURATION_MS_DEFAULT):
        """Initialize a FigureFactory

        Parameters
        ----------
        frame_duration_ms : float
            How long a single frame-to-frame transition lasts in the animation. If visualizing a
            single timestep (or you don't care how long a transition lasts), you can safely ignore
            this. This is helpful for visualizing data at the same rate it was recorded.
        """
        super().__init__(frame_duration_ms)

    def make_fig(self, show_before_return: bool = True, is_verbose: bool = False) -> go.Figure:
        """Constructs, returns, and optionally shows the plotly Figure"""
        if is_verbose:
            self.print_visualized_items()

        kwargs = dict()
        kwargs["layout"] = self._make_layout()
        kwargs["data"] = self._make_frame_data(0)

        # Adding the conditional here allows for visualization of a single timestep or multiple
        # timesteps.
        if self._num_frames > 1:
            kwargs["frames"] = self._make_animated_frames(kwargs["layout"])

        fig = go.Figure(**kwargs)

        if show_before_return:
            fig.update_layout(height=750)
            fig.show()

        return fig

    def _make_layout(self):
        """Constructs and returns the plotly figure layout"""
        scene = self._make_scene()
        updatemenus = make_update_menus(self._frame_duration_ms)
        sliders = make_sliders(self._num_frames)

        layout = go.Layout(scene=scene, updatemenus=updatemenus, sliders=[sliders], showlegend=True)

        return layout

    def _make_scene(self, buffer_abs: float = 0.1, buffer_rel: float = 0.0):
        """Constructs and returns the plotly visualization scene"""
        # We want to preserve aspect ratio so we manually set the axes here.
        xyz_mid = (self._scene_bounds_max + self._scene_bounds_min) / 2.
        xyz_half_ranges = (self._scene_bounds_max - self._scene_bounds_min) / 2.

        # Multiplying by 1 + buffer_rel to allow for relative buffers as well as absolute buffers in
        # the calculation of the visualized range.
        half_range = np.max(xyz_half_ranges) * (1.0 + buffer_rel)

        bounds_min = xyz_mid - half_range - buffer_abs
        bounds_max = xyz_mid + half_range + buffer_abs

        scene = go.layout.Scene(
            xaxis=go.layout.scene.XAxis(range=[bounds_min[0], bounds_max[0]], autorange=False),
            yaxis=go.layout.scene.YAxis(range=[bounds_min[1], bounds_max[1]], autorange=False),
            zaxis=go.layout.scene.ZAxis(range=[bounds_min[2], bounds_max[2]], autorange=False),
            # Setting aspectmode to 'cube' fixes issue with rescaling during animation. However, now
            # the view is not saved in the animation and is reset to initial scale whenever frames
            # are changed. This is annoying when zooming in and wanting to resume animation from the
            # zoomed view but this is less annoying than the rescaling.
            # TODO: Figure out how to update the scene when zooming.
            aspectmode='cube')
        return scene

    def _make_frame_data(self, frame_idx: int):
        """Constructs and returns the list necessary to make a plotly Frame"""
        frame_data = []

        for name, vertex_list in self._tracking_history.items():
            c: 'PointCloud' = vertex_list[frame_idx]
            kwargs = c.form_plotly_scatter3d_kwargs(marker_size=TRACKING_MARKER_SIZE,
                                                    trace_name=name,
                                                    markers_only=False)
            frame_data.append(go.Scatter3d(**kwargs))

        for name, cloud_list in self._clouds.items():
            c: 'PointCloud' = cloud_list[frame_idx]
            kwargs = c.form_plotly_scatter3d_kwargs(marker_size=POINT_CLOUD_MARKER_SIZE,
                                                    trace_name=name,
                                                    opacity=POINT_CLOUD_OPACITY)
            frame_data.append(go.Scatter3d(**kwargs))

        for mesh in self._mesh_vizs_static:
            kwargs = mesh.form_plotly_mesh3d_kwargs()
            frame_data.append(go.Mesh3d(**kwargs))

        for name, mesh_list in self._mesh_vizs_dynamic.items():
            m = mesh_list[frame_idx]
            kwargs = m.form_plotly_mesh3d_kwargs()
            frame_data.append(go.Mesh3d(**kwargs))

        return frame_data

    def _make_animated_frames(self, layout: go.Layout):
        """Constructs and returns the plotly Frames for the animation"""
        frames = []
        for i in range(self._num_frames):
            frame = go.Frame(data=self._make_frame_data(i), layout=layout, name=i)
            frames.append(frame)
        return frames
