# Importing the FigureFactory can cause a circular import with PointCloud unfortunately. There are
# ways to protect this but I believe it's also a code smell, so I should probably refactor to
# address this.
# from .figure_factory import FigureFactory
from .gripper_mesh_viz import GripperMeshViz
from .mesh_viz import MeshViz