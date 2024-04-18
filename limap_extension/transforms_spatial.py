import numpy as np
from scipy.spatial.transform import Rotation as R


def get_transform_matrix_from_pose_array(pose_arr: np.ndarray) -> np.ndarray:
    """Converts given 1D pose array (indexed from GT array) into 4x4 transformation matrix

    NOTE: This pose array is in the NED convention
    """
    pose_xyz = pose_arr[:3]
    pose_quat = pose_arr[3:]

    H = np.eye(4)
    H[:3, :3] = R.from_quat(pose_quat).as_matrix()
    H[:3, 3] = pose_xyz

    return H
