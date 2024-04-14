from pathlib import Path
import sys
from collections import namedtuple
from typing import Tuple

import numpy as np
from PIL import Image

CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parent

sys.path.append(REPO_ROOT.as_posix())
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array

DATA_DIR = CUR_DIR / "data" / "ocean" / "P006"

rgb_img_path_template = (DATA_DIR / "{idx:06d}_left.png").as_posix()
depth_img_path_template = (DATA_DIR / "{idx:06d}_left_depth.npy").as_posix()

TimestepData = namedtuple("TimestepData", ["rgb", "depth", "pose"])


def read_image_pair(idx_1: int = 310, idx_2: int = 311):
    """Reads the image pair for testing"""
    rgb_img_1 = np.array(Image.open(rgb_img_path_template.format(idx=idx_1)))
    depth_img_1 = np.load(depth_img_path_template.format(idx=idx_1))

    rgb_img_2 = np.array(Image.open(rgb_img_path_template.format(idx=idx_2)))
    depth_img_2 = np.load(depth_img_path_template.format(idx=idx_2))
    return rgb_img_1, depth_img_1, rgb_img_2, depth_img_2


def read_test_data(idx_1: int = 310, idx_2: int = 311) -> Tuple[TimestepData, TimestepData]:
    rgb_1, depth_1, rgb_2, depth_2 = read_image_pair(idx_1, idx_2)

    # Read pose at each index.
    pose_left_path = DATA_DIR / "pose_left.txt"
    pose_left = np.loadtxt(pose_left_path)
    pose_1 = get_transform_matrix_from_pose_array(pose_left[idx_1, :])
    pose_2 = get_transform_matrix_from_pose_array(pose_left[idx_2, :])

    td_1 = TimestepData(rgb_1, depth_1, pose_1)
    td_2 = TimestepData(rgb_2, depth_2, pose_2)
    return td_1, td_2
