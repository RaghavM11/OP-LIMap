from pathlib import Path
import sys

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
CACHED_CFG_PATH = ROOT_DIR / "cached_job.yml"


class Config:

    def __init__(self, job_cfg_path: Path = JOB_CFG_PATH):
        with job_cfg_path.open('r') as f:
            job_params = yaml.safe_load(f)
        self.scenario: str = job_params["scenario"]
        self.difficulty: str = job_params["difficulty"]
        self.trial: str = job_params["trial"]
        self.trial_path: Path = DATASET_DIR / self.scenario / self.difficulty / self.trial
        # self.img_direction = ImageDirection[job_params["img_direction"]]
        self.img_direction: ImageDirection = self._discern_img_direction(
            job_params["img_direction"])

        # TODO: Make this an enum in the projection based flow stuff
        self.method: str = job_params["method"]
        self.using_ground_truth_flow: bool = job_params["using_ground_truth_flow"]

    def _discern_img_direction(self, img_direction_str: str) -> ImageDirection:
        if img_direction_str == "left":
            return ImageDirection.LEFT
        elif img_direction_str == "right":
            return ImageDirection.RIGHT
        else:
            raise ValueError(
                f"Invalid image direction: {img_direction_str}, expected 'left' or 'right'")

    def cache_cfg(self):
        try:
            cfg_prev = Config(CACHED_CFG_PATH)
        except (FileNotFoundError, KeyError):
            # Key error is for when the new job config contains a new parameter the old one didn't.
            cfg_prev = None

        if not (cfg_prev == self):
            # Copy the current job config to the previous job config.
            with CACHED_CFG_PATH.open('w') as f:
                yaml.dump(
                    {
                        "scenario": self.scenario,
                        "difficulty": self.difficulty,
                        "trial": self.trial,
                        "img_direction": self.img_direction.value,
                        "method": self.method,
                        "using_ground_truth_flow": self.using_ground_truth_flow
                    }, f)

    def __eq__(self, other):
        if not isinstance(other, Config):
            return False
        return all(self.scenario == other.scenario, self.difficulty == other.difficulty,
                   self.trial == other.trial, self.img_direction == other.img_direction,
                   self.method == other.method)


def task_cache_cfg():

    def cache_cfg():
        cfg = Config()
        cfg.cache_cfg()

    return {
        "actions": [(cache_cfg, [])],
        "file_dep": [JOB_CFG_PATH],
        "targets": [CACHED_CFG_PATH],
    }


def idx_target_file_from_cfg(cfg: Config) -> Path:
    return ROOT_DIR / f"max_image_idx_{cfg.img_direction}.txt"


def frame_idx_to_str(frame_idx: int) -> str:
    return f"{frame_idx:06d}"


def get_mask_output_path(trial_path: Path, img_direction: ImageDirection, frame_idx_t: int) -> Path:
    f_t = frame_idx_to_str(frame_idx_t)
    out_dir = trial_path / f"dynamic_obj_masks_{img_direction.value}"
    return out_dir / f"{f_t}_{img_direction.value}_dynamic_obj_mask.png"


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


def task_process_masks():

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
                "file_dep": [CACHED_CFG_PATH],
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
