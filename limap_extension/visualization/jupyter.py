from typing import TYPE_CHECKING, Dict
from cdcpd_torch.visualization.plotly.visualizer import visualize_tracking_history as vth
from cdcpd_torch.visualization.plotly.util import FRAME_DURATION_MS_DEFAULT

if TYPE_CHECKING:
    import open3d as o3d

    from arm_clouds import PointCloudList


def visualize_tracking_history(tracked_vertex_dict: Dict[str, 'PointCloudList'],
                               num_frames: int,
                               point_clouds: 'PointCloudList' = None,
                               obstacle_meshes: Dict[str, 'o3d.geometry.TriangleMesh'] = dict(),
                               frame_duration_ms=FRAME_DURATION_MS_DEFAULT):
    """Visualizes the tracking history in jupyter-compatible way"""
    fig = vth(tracked_vertex_dict,
              num_frames,
              point_clouds,
              obstacle_meshes,
              frame_duration_ms,
              show_fig_before_return=False)
    fig.update_layout(height=750)
    fig.show()
    return fig
