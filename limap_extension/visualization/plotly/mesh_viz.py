from typing import Union
from pathlib import Path

import open3d as o3d
import numpy as np

from cdcpd_torch.visualization.plotly.util import MESH_OPACITY_DEFAULT


class MeshViz:
    name: str
    mesh: o3d.geometry.TriangleMesh

    def __init__(self, name: str, mesh_or_path: Union[o3d.geometry.TriangleMesh, Path]):
        self.name = name

        if isinstance(mesh_or_path, Path):
            self.mesh = o3d.io.read_triangle_mesh(mesh_or_path.as_posix())
        elif isinstance(mesh_or_path, o3d.geometry.TriangleMesh):
            self.mesh = mesh_or_path
        else:
            raise TypeError("Didn't understand `mesh_or_path` type:", type(mesh_or_path))

        # I don't think this is actually necessary.
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
        if not self.mesh.has_triangle_normals():
            self.mesh.compute_triangle_normals()

        self.tris = np.asarray(self.mesh.triangles)
        self.verts = np.asarray(self.mesh.vertices)

    def form_plotly_mesh3d_kwargs(self,
                                  opacity: float = MESH_OPACITY_DEFAULT,
                                  show_legend: bool = True):
        # TODO: Pull default mesh opacity from plotly utils.
        return {
            "x": self.verts[:, 0],
            "y": self.verts[:, 1],
            "z": self.verts[:, 2],
            "i": self.tris[:, 0],
            "j": self.tris[:, 1],
            "k": self.tris[:, 2],
            "opacity": opacity,
            "name": self.name,
            "showlegend": show_legend
        }

    def get_min_bounds(self):
        return np.min(self.verts, axis=0)

    def get_max_bounds(self):
        return np.max(self.verts, axis=0)