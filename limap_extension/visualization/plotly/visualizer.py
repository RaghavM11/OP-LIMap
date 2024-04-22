"""An awful implementation of low-level plotly visualization for tracking over time"""
from typing import Dict, TYPE_CHECKING
from warnings import warn

import numpy as np
import plotly.graph_objects as go

from limap_extension.point_cloud_list import PointCloudList
from limap_extension.visualization.plotly.util import (make_sliders, make_update_menus,
                                                       FRAME_DURATION_MS_DEFAULT,
                                                       TRACKING_MARKER_SIZE, POINT_CLOUD_OPACITY,
                                                       POINT_CLOUD_MARKER_SIZE,
                                                       MESH_OPACITY_DEFAULT)
from limap_extension.visualization.plotly.mesh_viz import MeshViz

import open3d as o3d

if TYPE_CHECKING:
    from limap_extension.point_cloud import PointCloud


# TODO: Refactor to a class and have a method in the class for adding multiple tracked
# configurations.
# TODO: Make point cloud input a dictionary as well so we can show input cloud and segmentation
# side-by-side
def visualize_tracking_history(tracked_vertex_dict: Dict[str, PointCloudList],
                               num_frames: int,
                               point_clouds: PointCloudList = None,
                               obstacle_meshes: Dict[str, o3d.geometry.TriangleMesh] = dict(),
                               frame_duration_ms=FRAME_DURATION_MS_DEFAULT,
                               show_fig_before_return: bool = True) -> go.Figure:
    """This absolute mess of a function visualizes tracked vertices over time

    NOTE: This assumes a rope. If trying to visualize a cloth, I'll have to adapt this function to
    show the edges specified instead of inferring the edges.
    """
    # Opting not to use the DeprecationWarning category as the default warning filter in Python
    # doesn't output this to terminal.
    warn("Use FigureFactory class instead. visualize_tracking_history is no longer supported.")

    _prep_verts(tracked_vertex_dict)
    _prep_clouds(point_clouds)
    prepped_meshes = _prep_obstacle_meshes(obstacle_meshes)

    layout = _make_layout(tracked_vertex_dict, num_frames, point_clouds, prepped_meshes,
                          frame_duration_ms)
    frames = _make_animation_frames(tracked_vertex_dict, num_frames, layout, point_clouds,
                                    prepped_meshes)
    fig_data = _make_initial_plot_data(tracked_vertex_dict, point_clouds, prepped_meshes)

    fig = go.Figure(data=fig_data, layout=layout, frames=frames)

    if show_fig_before_return:
        fig.show()

    return fig


def _prep_verts(tracked_vertex_dict: Dict[str, PointCloudList]) -> None:
    for verts_list in tracked_vertex_dict.values():
        verts_list.to_numpy()

        verts = verts_list[0].xyz
        if verts.shape[1] == 3 and verts.shape[0] > 3:
            raise RuntimeError("Detected vertices are in (time, num_points, XYZ) shape!")

        verts_list.normalize_rgb()


def _prep_clouds(point_clouds_list: PointCloudList = None):
    if point_clouds_list:
        point_clouds_list.to_numpy()
        point_clouds_list.normalize_rgb()


def _prep_obstacle_meshes(obstacle_meshes: Dict[str, o3d.geometry.TriangleMesh] = dict()) -> Dict[
        str, MeshViz]:
    """Construct `MeshViz` objects for all meshes to be visualized"""
    mesh_vizes = dict()
    for name, mesh in obstacle_meshes.items():
        mesh_vizes[name] = MeshViz(name, mesh)
    return mesh_vizes


def _make_animation_frames(template_verts: Dict[str, PointCloudList],
                           num_frames,
                           layout,
                           clouds: PointCloudList = None,
                           prepped_meshes: Dict[str, MeshViz] = None):
    frames = []
    for i in range(num_frames):
        data = []

        # Add all tracking history (e.g. CPD output, post-processing output)
        for name, cloud_list in template_verts.items():
            c = cloud_list[i]
            template_scatter_kwargs = c.form_plotly_scatter3d_kwargs(
                marker_size=TRACKING_MARKER_SIZE, trace_name=name, markers_only=False)
            data.append(go.Scatter3d(**template_scatter_kwargs))

        if clouds:
            c: 'PointCloud' = clouds[i]
            cloud_kwargs = c.form_plotly_scatter3d_kwargs(marker_size=POINT_CLOUD_MARKER_SIZE,
                                                          trace_name="Point Cloud",
                                                          opacity=POINT_CLOUD_OPACITY)
            data.append(go.Scatter3d(**cloud_kwargs))

        if prepped_meshes:
            for name, mv in prepped_meshes.items():
                # TODO: Just use mv.form_plotly_mesh3d_kwargs
                # data.append(go.Mesh3d(**mv.form_plotly_mesh3d_kwargs()))
                data.append(
                    go.Mesh3d(x=mv.verts[:, 0],
                              y=mv.verts[:, 1],
                              z=mv.verts[:, 2],
                              i=mv.tris[:, 0],
                              j=mv.tris[:, 1],
                              k=mv.tris[:, 2],
                              opacity=MESH_OPACITY_DEFAULT,
                              showlegend=True))

        frame = go.Frame(data=data, layout=layout, name=i)
        frames.append(frame)
    return frames


def _make_initial_plot_data(template_verts: Dict[str, PointCloudList],
                            clouds: PointCloudList = None,
                            prepped_meshes: Dict[str, MeshViz] = None):
    fig_data = []

    # Add initial frame for all tracking histories (e.g. CPD output, post-processing output)
    # TODO: Add a method to PointCloud class for making a plotly.graph_objects.Scatter3d trace and
    # call that method here to reduce repeated code.
    for name, vertex_list in template_verts.items():
        c = vertex_list[0]
        template_scatter_kwargs = c.form_plotly_scatter3d_kwargs(marker_size=TRACKING_MARKER_SIZE,
                                                                 trace_name=name,
                                                                 markers_only=False)
        fig_data.append(go.Scatter3d(**template_scatter_kwargs))

    if clouds:
        c: 'PointCloud' = clouds[0]
        cloud_kwargs = c.form_plotly_scatter3d_kwargs(marker_size=POINT_CLOUD_MARKER_SIZE,
                                                      trace_name="Point Cloud",
                                                      opacity=POINT_CLOUD_OPACITY)
        fig_data.append(go.Scatter3d(**cloud_kwargs))

    if prepped_meshes:
        for name, mv in prepped_meshes.items():
            # TODO: Just use mv.form_plotly_mesh3d_kwargs()
            # fig_data.append(go.Mesh3d(**mv.form_plotly_mesh3d_kwargs()))
            fig_data.append(
                go.Mesh3d(x=mv.verts[:, 0],
                          y=mv.verts[:, 1],
                          z=mv.verts[:, 2],
                          i=mv.tris[:, 0],
                          j=mv.tris[:, 1],
                          k=mv.tris[:, 2],
                          opacity=MESH_OPACITY_DEFAULT,
                          name=name))

    return fig_data


def _make_scene(template_verts: Dict[str, PointCloudList],
                clouds: PointCloudList = None,
                prepped_meshes: Dict[str, MeshViz] = None,
                buffer=0.1):
    """Essentially just calculates the axis ranges for the plots

    NOTE: Ignoring obstacle meshes in range calculation for now
    """

    xyz_min = 1e15 * np.ones(3)
    xyz_max = -1e15 * np.ones(3)

    frame_lengths = [len(v) for v in template_verts.values()]
    if clouds is not None:
        frame_lengths.append(len(clouds))

    num_frames = min(frame_lengths)

    for i in range(num_frames):
        verts_i = [verts_list[i].xyz for verts_list in template_verts.values()]
        if clouds:
            c_i = clouds[i]
            all_xyz_vals_i = verts_i + [c_i.xyz]
        else:
            all_xyz_vals_i = verts_i

        xyz_all_min_i = [np.min(v, axis=1) for v in all_xyz_vals_i] + [xyz_min]
        xyz_all_max_i = [np.max(v, axis=1) for v in all_xyz_vals_i] + [xyz_max]

        xyz_min = np.min(np.stack(xyz_all_min_i, axis=1), axis=1)
        xyz_max = np.max(np.stack(xyz_all_max_i, axis=1), axis=1)

    # We want to preserve aspect ratio so we manually set the axes here.
    xyz_mids = (xyz_max + xyz_min) / 2.
    xyz_half_ranges = (xyz_max - xyz_min) / 2.
    half_range = np.max(xyz_half_ranges)
    xyz_range_mins = xyz_mids - half_range - buffer
    xyz_range_maxes = xyz_mids + half_range + buffer

    scene = go.layout.Scene(
        xaxis=go.layout.scene.XAxis(range=[xyz_range_mins[0], xyz_range_maxes[0]], autorange=False),
        yaxis=go.layout.scene.YAxis(range=[xyz_range_mins[1], xyz_range_maxes[1]], autorange=False),
        zaxis=go.layout.scene.ZAxis(range=[xyz_range_mins[2], xyz_range_maxes[2]], autorange=False),
        # Setting aspectmode to 'cube' fixes issue with rescaling during animation. However, now the
        # view is not saved in the animation and is reset to initial scale whenever frames are
        # changed. This is annoying when zooming in and wanting to resume animation from the zoomed
        # view but this is less annoying than the rescaling.
        # TODO: Figure out how to update the scene when zooming.
        aspectmode='cube')
    return scene


def _make_layout(template_verts: Dict[str, PointCloudList],
                 num_frames: int,
                 clouds: PointCloudList = None,
                 prepped_meshes: Dict[str, MeshViz] = None,
                 frame_duration_ms: float = FRAME_DURATION_MS_DEFAULT):
    scene = _make_scene(template_verts, clouds, prepped_meshes)
    updatemenus = make_update_menus(frame_duration_ms)
    sliders = make_sliders(num_frames)

    layout = go.Layout(scene=scene, updatemenus=updatemenus, sliders=[sliders], showlegend=True)

    return layout
