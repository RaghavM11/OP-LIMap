from typing import TYPE_CHECKING, Optional
import time

import rerun as rr
import numpy as np

from limap_extension.visualization.figure_factory_base import FigureFactoryBase
# from limap_extension.visualization.plotly.util import FRAME_DURATION_MS_DEFAULT
from limap_extension.visualization.rerun.rr_util import log_def_obj_config
# from limap_extension.data_utils.stopwatch import Stopwatch
FRAME_DURATION_MS_DEFAULT = 50

# if TYPE_CHECKING:
#     from arm_clouds import PointCloud

RR_TIMELINE_NAME = "stable_time"
MS_TO_SECONDS = 1e-3
TRACKING_MARKER_SIZE = 0.01
POINT_CLOUD_MARKER_SIZE = 0.005
TRACKING_MARKER_COLOR = np.array((1.0, 0.0, 0.0))
POINT_CLOUD_MARKER_COLOR = np.array((0.0, 0.0, 1.0))


class FigureFactory(FigureFactoryBase):

    def __init__(self, frame_duration_ms: float = FRAME_DURATION_MS_DEFAULT):
        super().__init__(frame_duration_ms)

    def make_fig(self,
                 is_using_browser: bool = False,
                 is_verbose: bool = False,
                 t_min_secs: Optional[bool] = None):
        """Initializes the rerun viewer and logs the frames

        NOTE: Rerun will drop messages if the Python script finishes (thus, shutting down the
        server) before the viewer is loaded in the browser. This is why we have a sleep at the end.
        """
        # s = Stopwatch("Rerun Figure Creation", is_printing=is_verbose)
        spawn = not is_using_browser
        rr.init("CDCPD_torch_visualization", spawn=spawn)
        if is_using_browser:
            rr.serve()

        rr.set_time_seconds(RR_TIMELINE_NAME, 0)

        self._make_frames()
        # t_elapsed = s.stop()

        # Sleep to ensure the viewer has time to load before the server shuts down. This isn't a
        # problem when using the desktop app viewer as the server persists upon completion of the
        # Python process calling this function.
        # if is_using_browser:
        #     t_required = t_min_secs if t_min_secs is not None else 5.0
        #     t_delta = t_required - t_elapsed
        #     if t_delta > 0:
        #         time.sleep(t_delta)

    def _make_single_frame(self, frame_idx: int):
        rr.set_time_seconds(RR_TIMELINE_NAME, frame_idx * self._frame_duration_ms * MS_TO_SECONDS)

        for name, vertex_list in self._tracking_history.items():
            c: 'PointCloud' = vertex_list[frame_idx].copy()
            kwargs = c.form_rerun_kwargs(radii=TRACKING_MARKER_SIZE, color=TRACKING_MARKER_COLOR)
            log_def_obj_config(name, kwargs.pop("positions"), self._tracking_edges[name], **kwargs)

        for name, cloud_list in self._clouds.items():
            c: 'PointCloud' = cloud_list[frame_idx].copy()
            rr.log(
                name,
                rr.Points3D(**c.form_rerun_kwargs(radii=POINT_CLOUD_MARKER_SIZE,
                                                  color=POINT_CLOUD_MARKER_COLOR)))

        for name, mesh_list in self._mesh_vizs_dynamic.items():
            m = mesh_list[frame_idx]
            # kwargs = m.form_plotly_mesh3d_kwargs()
            # frame_data.append(go.Mesh3d(**kwargs))
            rr.log(
                name,
                rr.Mesh3D(vertex_positions=m.verts,
                          indices=m.tris,
                          vertex_colors=(0.0, 1.0, 0.0, 0.5)))

    def _make_frames(self):
        # Log static meshes since they don't change and continue to display with one log call in
        # rerun.
        for mesh in self._mesh_vizs_static:
            rr.log(mesh.name, rr.Mesh3D(vertex_positions=mesh.verts, indices=mesh.tris))

        for i in range(self._num_frames):
            self._make_single_frame(i)
