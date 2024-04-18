from pathlib import Path
import sys
# import pickle as pkl
from typing import Dict

import numpy as np
from PIL import Image
import yaml

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parents[1]
sys.path.append(REPO_DIR.as_posix())

from limap_extension.utils.io import read_rgbd, read_pose
from limap_extension.constants import ImageDirection
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array
from limap_extension.optical_flow import OpticalFlow, Args, RAFT_MODEL_PATH
from limap_extension.projection_based_flow import projection_based_motion_segmentation

# TODO: Put all of this stuff into config files that are read in by doit.
DATASET_DIR = REPO_DIR / "datasets"
JOB_CFG_PATH = ROOT_DIR / "job.yml"

# # SCENARIO = "ocean"
# SCENARIO = "carwelding"
# DIFFICULTY = "Hard"
# # TRIAL = "P006"
# TRIAL = "P001"


class Config:

    def __init__(self):
        with JOB_CFG_PATH.open('r') as f:
            job_params = yaml.safe_load(f)
        self.scenario = job_params["scenario"]
        self.difficulty = job_params["difficulty"]
        self.trial = job_params["trial"]
        self.trial_path = DATASET_DIR / self.scenario / self.difficulty / self.trial
        # self.img_direction = ImageDirection[job_params["img_direction"]]
        self.img_direction = self._discern_img_direction(job_params["img_direction"])

    def _discern_img_direction(self, img_direction_str: str) -> ImageDirection:
        if img_direction_str == "left":
            return ImageDirection.LEFT
        elif img_direction_str == "right":
            return ImageDirection.RIGHT
        else:
            raise ValueError(
                f"Invalid image direction: {img_direction_str}, expected 'left' or 'right'")


# def read_job_params() -> Dict:
#     with JOB_CFG_PATH.open('r') as f:
#         job_params = yaml.safe_load(f)
#     return job_params

# def trial_path_from_cfg(cfg: Dict) -> Path:
#     scenario = cfg["scenario"]
#     difficulty = cfg["difficulty"]
#     trial = cfg["trial"]
#     return DATASET_DIR / scenario / difficulty / trial


def idx_target_file_from_cfg(cfg: Config) -> Path:
    return ROOT_DIR / f"max_image_idx_{cfg.img_direction}.txt"


# TARGET_LIST_RIGHT = ROOT_DIR / "file_list_right.pkl"
# TARGET_LIST_LEFT = ROOT_DIR / "file_list_left.pkl"
# TARGET_LIST_LEFT = ROOT_DIR / "max_image_idx_left.txt"

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


def frame_idx_to_str(frame_idx: int) -> str:
    return f"{frame_idx:06d}"


def get_mask_output_path(trial_path: Path, img_direction: ImageDirection, frame_idx_t: int) -> Path:
    f_t = frame_idx_to_str(frame_idx_t)
    out_dir = trial_path / f"dynamic_obj_masks_{img_direction.value}"
    return out_dir / f"{f_t}_{img_direction.value}_dynamic_obj_mask.png"


# def targets_from_file_list(direction: ImageDirection):
#     if direction == ImageDirection.RIGHT:
#         # file_list_path = TARGET_LIST_RIGHT
#         raise NotImplementedError(
#             "Right image direction not implemented due to no ground truth flow.")
#     elif direction == ImageDirection.LEFT:
#         file_list_path = TARGET_LIST_LEFT

#     with open(file_list_path, 'rb') as f:
#         file_list = pkl.load(f)

#     targets = []
#     for (i, j) in file_list:
#         i = int(i)
#         j = int(j)
#         targets.append(get_mask_output_path(TRIAL_PATH, direction, i, j))

#     return targets


def save_mask(mask: np.ndarray, out_path: Path):
    mask = mask.astype(np.uint8) * 255
    mask = Image.fromarray(mask)
    mask.save(out_path)


def get_max_img_idx(trial_path: Path, img_direction: ImageDirection) -> int:
    img_gt_dir = trial_path / f"image_{img_direction.value}"
    img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
    max_idx = max(img_idxs)
    return max_idx


def process_single_frame_pair(trial_path: Path, img_direction: ImageDirection, frame_idx: int,
                              poses: np.ndarray):
    i = frame_idx - 1
    j = frame_idx

    out_path = get_mask_output_path(trial_path, img_direction, j)

    (rgb_2, depth_2) = read_rgbd(trial_path, img_direction, j)
    if frame_idx == 0:
        # No previous frame to compare to, so we indicate that nothing is dynamic.
        save_mask(np.zeros_like(depth_2), out_path)
        return

    (rgb_1, depth_1) = read_rgbd(trial_path, img_direction, i)

    poses = read_pose(trial_path, img_direction)
    pose_1 = get_transform_matrix_from_pose_array(poses[i, :])
    pose_2 = get_transform_matrix_from_pose_array(poses[j, :])

    flow = OpticalFlow(None)
    flow.load_model(RAFT_MODEL_PATH, Args())

    mask = projection_based_motion_segmentation(rgb_1, depth_1, rgb_2, depth_2, pose_1, pose_2,
                                                flow)

    save_mask(mask, out_path)


# def task_generate_file_list_right():

#     def generate_file_list_right():
#         img_gt_dir = TRIAL_PATH / "image_right"
#         img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
#         max_idx = max(img_idxs)
#         # print("max_idx: ", max_idx)

#         # img_name_template = "{idx:06d}_right.png"
#         frame_idx_template = "{idx:06d}"

#         # poses = read_pose(TRIAL_PATH, ImageDirection.RIGHT)
#         idxs = list(range(max_idx))
#         idx_pairs = [(frame_idx_template.format(idx=idxs[i]),
#                       frame_idx_template.format(idx=idxs[i + 1])) for i in range(max_idx - 1)]

#         # for i in range(max_idx - 1):
#         #     # process_single_frame_pair(ImageDirection.RIGHT, i, None)
#         #     # this doesn't do the actual processing, it just defines the tasks.
#         #     name_i = frame_idx_template.format(idx=i)
#         #     name_j = frame_idx_template.format(idx=i + 1)

#         #     targets = get_img_bbox_paths(TRIAL_PATH, ImageDirection.RIGHT, i, i + 1)

#         with open(TARGET_LIST_RIGHT, 'wb') as f:
#             pkl.dump(idx_pairs, f)

#     return {
#         "actions": [(generate_file_list_right, [])],
#         "targets": [TARGET_LIST_RIGHT],
#     }

# def task_process_right():
#     img_gt_dir = TRIAL_PATH / "image_right"
#     img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
#     max_idx = max(img_idxs)
#     # print("max_idx: ", max_idx)

#     # img_name_template = "{idx:06d}_right.png"
#     frame_idx_template = "{idx:06d}"

#     poses = read_pose(TRIAL_PATH, ImageDirection.RIGHT)

#     for i in range(max_idx - 1):
#         # process_single_frame_pair(ImageDirection.RIGHT, i, None)
#         # this doesn't do the actual processing, it just defines the tasks.
#         name_i = frame_idx_template.format(idx=i)
#         name_j = frame_idx_template.format(idx=i + 1)

#         targets = get_img_bbox_paths(TRIAL_PATH, ImageDirection.RIGHT, i, i + 1)
#         # print("Targets:", targets)

#         yield {
#             "basename": f"process_right_{name_i}_{name_j}",
#             "actions": [(process_single_frame_pair, [ImageDirection.RIGHT, i, poses])],
#             "file_dep": [TARGET_LIST_RIGHT],
#             "targets": targets,
#         }

# def task_generate_idx_file():

#     cfg = read_job_params()
#     target_file = idx_target_file_from_cfg(cfg)

#     def generate_file_idx_left():
#         trial_path = trial_path_from_cfg(cfg)
#         img_gt_dir = trial_path / "image_left"
#         img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
#         max_idx = max(img_idxs)

#         target_file.write_text(f"{max_idx}\n")

#     return {
#         "actions": [(generate_file_idx_left, [])],
#         "targets": [target_file],
#         "file_dep": [JOB_CFG_PATH]
#     }


def task_process_masks():

    # def get_max_idx(idx_file: Path):
    #     with idx_file.open('r'):
    #         max_idx = int(idx_file.readline())
    #     return max_idx

    # yield {
    #     "basename": "read_max_idx_left",
    #     "actions": [],
    #     "file_dep": [TARGET_LIST_LEFT],
    # }

    cfg = Config()
    max_idx = get_max_img_idx(cfg.trial_path, cfg.img_direction)

    # Get a dummy mask output path to make that directory.
    dummy_out_path = get_mask_output_path(cfg.trial_path, cfg.img_direction, 0)
    dummy_out_path.parent.mkdir(exist_ok=True)

    def define_jobs():
        poses = read_pose(cfg.trial_path, cfg.img_direction)

        for idx_t in range(max_idx + 1):
            # this doesn't do the actual processing, it just defines the tasks.
            f_t = frame_idx_to_str(idx_t)
            targets = get_mask_output_path(cfg.trial_path, cfg.img_direction, idx_t)
            # print("targets: ", targets)

            yield {
                "basename":
                f"process_mask_{f_t}",
                "actions":
                [(process_single_frame_pair, [cfg.trial_path, cfg.img_direction, idx_t, poses])],
                "file_dep": [JOB_CFG_PATH],
                "targets": [targets],
            }

    yield define_jobs()


# def task_zip_reprojections():

#     def get_zip_target(direction: ImageDirection):
#         return DATASET_DIR / f"{SCENARIO}_{DIFFICULTY}_{TRIAL}_reprojections_{direction.value}.zip"

#     def zip_reprojections(direction: ImageDirection):
#         import zipfile

#         # out_path = DATASET_DIR /
#         # f"{SCENARIO}_{DIFFICULTY}_{TRIAL}_reprojections_{direction.value}.zip"
#         out_path = get_zip_target(direction)
#         # in_dir = TRIAL_PATH / f"reproj_{direction.value}"

#         paths_to_zip = targets_from_file_list(direction)

#         with zipfile.ZipFile(out_path.as_posix(), 'w') as z:
#             # Want to write the zip file in a way that is easy to extract, so chop off everything
#             # prior to the scenario.
#             for path_pair in paths_to_zip:
#                 for p in path_pair:
#                     # arc_name = out_path.relative_to(DATASET_DIR)
#                     arc_name = p.relative_to(DATASET_DIR)
#                     z.write(p, arc_name)

#     for direction in [ImageDirection.RIGHT, ImageDirection.LEFT]:
#         file_dep = [TARGET_LIST_RIGHT] if direction == ImageDirection.RIGHT else [TARGET_LIST_LEFT]
#         zip_target = get_zip_target(direction)
#         yield {
#             "basename": f"zip_reprojections_{direction.value}",
#             "actions": [(zip_reprojections, [direction])],
#             "file_dep": file_dep,
#             "targets": [zip_target],
#         }

if __name__ == "__main__":
    # img_gt_dir = TRIAL_PATH / "image_right"
    # img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
    # max_idx = max(img_idxs)
    # print("max_idx: ", max_idx)

    # # img_name_template = "{idx:06d}_right.png"

    # for i in range(max_idx - 1):
    #     process_single_frame_pair(ImageDirection.RIGHT, i, None)
    # #     print(f"img_path: {img_path}")

    # print(targets_from_file_list(ImageDirection.RIGHT))
    pass
