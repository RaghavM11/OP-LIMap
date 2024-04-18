from typing import TYPE_CHECKING, Optional

import open3d as o3d
import numpy as np

from cdcpd_torch.visualization.plotly.mesh_viz import MeshViz

if TYPE_CHECKING:
    from cdcpd_torch.data_utils.types.grippers import GripperInfoSingle

GRIPPER_MESH_KWARGS = {
    "cylinder_radius": 0.025,
    "cone_radius": 0.035,
    "cylinder_height": 0.1,
    "cone_height": 0.08,
    "resolution": 10,
    "cylinder_split": 1
}

GRIPPER_MESH_SIMPLE = o3d.geometry.TriangleMesh.create_arrow(**GRIPPER_MESH_KWARGS)
# Translate the arrow such that the tip of the cone is located at the origin.
GRIPPER_MESH_SIMPLE.translate(
    (0.0, 0.0, -(GRIPPER_MESH_KWARGS["cylinder_height"] + GRIPPER_MESH_KWARGS["cone_height"])))
# Now rotate the arrow such that it is pointing in the direction of the x-axis.
GRIPPER_MESH_SIMPLE.rotate(np.array(((0, 0, 1), (0, 1, 0), (1, 0, 0))), (0, 0, 0))


class GripperMeshViz(MeshViz):

    def __init__(self,
                 name: str,
                 gripper_info: 'GripperInfoSingle',
                 gripper_mesh_orig: Optional[o3d.geometry.TriangleMesh] = None,
                 add_axis_to_mesh: bool = False):
        """Initialize a `GripperMeshViz` object

        NOTE: `gripper_mesh_orig` is a mesh representing your gripper, aligned with the x-axis,
        where the origin corresponds to where the gripper would grasp vertices.
        """
        if gripper_mesh_orig is None:
            gripper_mesh_orig = GRIPPER_MESH_SIMPLE

        mesh = self._prep_mesh(gripper_mesh_orig, gripper_info.q, add_axis_to_mesh)

        # Now that we have the mesh, we can instantiate the base class
        super().__init__(name, mesh)

    def _prep_mesh(self,
                   mesh_orig: o3d.geometry.TriangleMesh,
                   gripper_pose: 'GripperSinglePose',
                   add_axis_to_mesh: bool = False):
        # Copy the untranslated/unrotated original mesh so that the transformations we apply don't
        # modify it.
        mesh = o3d.geometry.TriangleMesh(mesh_orig)

        if add_axis_to_mesh:
            mesh += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

        self._rotate_mesh_to_match_gripper_info(mesh, gripper_pose)
        self._translate_mesh_to_match_gripper_info(mesh, gripper_pose)

        return mesh

    def _rotate_mesh_to_match_gripper_info(self, mesh: o3d.geometry.TriangleMesh,
                                           g: 'GripperSinglePose'):
        g_rot_mat = g.rotation()
        mesh.rotate(g_rot_mat, (0, 0, 0))

    def _translate_mesh_to_match_gripper_info(self, mesh: o3d.geometry.TriangleMesh,
                                              g: 'GripperSinglePose'):
        mesh.translate(g.translation())
