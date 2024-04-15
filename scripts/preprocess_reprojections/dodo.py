from pathlib import Path
import sys
import pickle as pkl

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parents[1]
sys.path.append(REPO_DIR.as_posix())
# print("Repo root:", REPO_DIR)
# quit()

from preprocess_reprojections import (save_reprojection, read_rgbd, ImageType, ImageDirection,
                                      read_pose, get_img_bbox_paths)
from limap_extension.img_cloud_transforms import reproject_img
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array

DATASET_DIR = REPO_DIR / "datasets"

SCENARIO = "ocean"
DIFFICULTY = "Hard"
TRIAL = "P006"

TRIAL_PATH = DATASET_DIR / SCENARIO / DIFFICULTY / TRIAL

TARGET_LIST_RIGHT = ROOT_DIR / "file_list_right.pkl"
TARGET_LIST_LEFT = ROOT_DIR / "file_list_left.pkl"

# def task_make_file_list():
#     """Make a list of all the files in the dataset directory."""
#     dataset_dir = REPO_ROOT / "datasets"
#     file_list_path = dataset_dir / "file_list.txt"

#     return {
#         "actions": [
#             f"ls -R {dataset_dir} > {file_list_path}"
#         ],
#         "file_dep": [REPO_ROOT / "scripts/preprocess_reprojections/dodo.py"],
#         "targets": [file_list_path],
#     }


def process_single_frame_pair(img_direction: ImageDirection, frame_start: int, poses: np.ndarray):
    i = frame_start
    j = frame_start + 1

    (rgb_1, depth_1) = read_rgbd(TRIAL_PATH, img_direction, i)
    # (rgb_2, depth_2) = read_rgbd(TRIAL_PATH, img_direction, j)

    poses = read_pose(TRIAL_PATH, img_direction)
    pose_1 = get_transform_matrix_from_pose_array(poses[i, :])
    pose_2 = get_transform_matrix_from_pose_array(poses[j, :])

    img_1_in_frame_2, valid_bbox = reproject_img(rgb_1,
                                                 depth_1,
                                                 pose_1,
                                                 pose_2,
                                                 interpolation_method="clough_tocher")

    save_reprojection(TRIAL_PATH, img_1_in_frame_2, valid_bbox, img_direction, i, i + 1)


def task_generate_file_list_right():

    def generate_file_list_right():
        img_gt_dir = TRIAL_PATH / "image_right"
        img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
        max_idx = max(img_idxs)
        # print("max_idx: ", max_idx)

        # img_name_template = "{idx:06d}_right.png"
        frame_idx_template = "{idx:06d}"

        # poses = read_pose(TRIAL_PATH, ImageDirection.RIGHT)
        idxs = list(range(max_idx))
        idx_pairs = [(frame_idx_template.format(idx=idxs[i]),
                      frame_idx_template.format(idx=idxs[i + 1])) for i in range(max_idx - 1)]

        # for i in range(max_idx - 1):
        #     # process_single_frame_pair(ImageDirection.RIGHT, i, None)
        #     # this doesn't do the actual processing, it just defines the tasks.
        #     name_i = frame_idx_template.format(idx=i)
        #     name_j = frame_idx_template.format(idx=i + 1)

        #     targets = get_img_bbox_paths(TRIAL_PATH, ImageDirection.RIGHT, i, i + 1)

        with open(TARGET_LIST_RIGHT, 'wb') as f:
            pkl.dump(idx_pairs, f)

    return {
        "actions": [(generate_file_list_right, [])],
        "targets": [TARGET_LIST_RIGHT],
    }


def task_process_right():
    img_gt_dir = TRIAL_PATH / "image_right"
    img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
    max_idx = max(img_idxs)
    # print("max_idx: ", max_idx)

    # img_name_template = "{idx:06d}_right.png"
    frame_idx_template = "{idx:06d}"

    poses = read_pose(TRIAL_PATH, ImageDirection.RIGHT)

    for i in range(max_idx - 1):
        # process_single_frame_pair(ImageDirection.RIGHT, i, None)
        # this doesn't do the actual processing, it just defines the tasks.
        name_i = frame_idx_template.format(idx=i)
        name_j = frame_idx_template.format(idx=i + 1)

        targets = get_img_bbox_paths(TRIAL_PATH, ImageDirection.RIGHT, i, i + 1)
        # print("Targets:", targets)

        yield {
            "basename": f"process_right_{name_i}_{name_j}",
            "actions": [(process_single_frame_pair, [ImageDirection.RIGHT, i, poses])],
            "file_dep": [TARGET_LIST_RIGHT],
            "targets": targets,
        }


def task_generate_file_list_left():

    def generate_file_list_left():
        img_gt_dir = TRIAL_PATH / "image_left"
        img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
        max_idx = max(img_idxs)

        frame_idx_template = "{idx:06d}"

        idxs = list(range(max_idx))
        idx_pairs = [(frame_idx_template.format(idx=idxs[i]),
                      frame_idx_template.format(idx=idxs[i + 1])) for i in range(max_idx - 1)]

        with open(TARGET_LIST_LEFT, 'wb') as f:
            pkl.dump(idx_pairs, f)

    return {
        "actions": [(generate_file_list_left, [])],
        "targets": [TARGET_LIST_LEFT],
    }


def task_process_left():
    img_gt_dir = TRIAL_PATH / "image_left"
    img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
    max_idx = max(img_idxs)
    frame_idx_template = "{idx:06d}"

    poses = read_pose(TRIAL_PATH, ImageDirection.LEFT)

    for i in range(max_idx - 1):
        # this doesn't do the actual processing, it just defines the tasks.
        name_i = frame_idx_template.format(idx=i)
        name_j = frame_idx_template.format(idx=i + 1)

        targets = get_img_bbox_paths(TRIAL_PATH, ImageDirection.LEFT, i, i + 1)
        # print("Targets:", targets)

        yield {
            "basename": f"process_left_{name_i}_{name_j}",
            "actions": [(process_single_frame_pair, [ImageDirection.LEFT, i, poses])],
            "file_dep": [TARGET_LIST_LEFT],
            "targets": targets,
        }


if __name__ == "__main__":
    img_gt_dir = TRIAL_PATH / "image_right"
    img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
    max_idx = max(img_idxs)
    print("max_idx: ", max_idx)

    # img_name_template = "{idx:06d}_right.png"

    for i in range(max_idx - 1):
        process_single_frame_pair(ImageDirection.RIGHT, i, None)
    #     print(f"img_path: {img_path}")
