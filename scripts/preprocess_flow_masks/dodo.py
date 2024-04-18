from pathlib import Path
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass
from copy import deepcopy
import pickle as pkl

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
# CACHED_JOBS_PATH = ROOT_DIR / "cached_jobs.yml"
CACHED_JOBS_DIR = ROOT_DIR / "cached_jobs"
CACHED_JOBS_DIR.mkdir(exist_ok=True, parents=True)

# @dataclass(frozen=True)
# class Job:
#     scenario: str
#     difficulty: str
#     trial: str
#     trial_path: Path
#     img_direction: ImageDirection
#     method: str
#     using_ground_truth_flow: bool


@dataclass(frozen=True, eq=True)
class Job:
    name: str
    scenario: str
    difficulty: str
    trial: str
    img_direction: ImageDirection
    method: str
    using_ground_truth_flow: bool
    trial_path: Path

    @staticmethod
    def from_cfg_dict(cfg: Dict) -> 'Job':
        trial_path = DATASET_DIR / cfg["scenario"] / cfg["difficulty"] / cfg["trial"]
        cfg["img_direction"] = (ImageDirection.LEFT
                                if cfg["img_direction"] == "left" else ImageDirection.RIGHT)
        return Job(**cfg, trial_path=trial_path)  #, trial_path=trial_path)

    def get_cache_path(self) -> Path:
        return CACHED_JOBS_DIR / f"{self.name}.pkl"

    def cache_job(self) -> Path:
        out_path = self.get_cache_path()
        if out_path.exists():
            return out_path

        # with out_path.open('w') as f:
        #     yaml.dump(vars(self), f)
        with out_path.open('wb') as f:
            pkl.dump(self, f)
        return out_path


# def job_from_cfg_dict(cfg: Dict) -> Job:
#     return Job(scenario=cfg["scenario"],
#                difficulty=cfg["difficulty"],
#                trial=cfg["trial"],
#                img_direction=cfg["img_direction"],
#                method=cfg["method"],
#                using_ground_truth_flow=cfg["using_ground_truth_flow"])

# class Job:

#     def __init__(self, job_params: dict):
#         self._job_params = job_params
#         self.scenario: str = job_params["scenario"]
#         self.difficulty: str = job_params["difficulty"]
#         self.trial: str = job_params["trial"]
#         self.trial_path: Path = DATASET_DIR / self.scenario / self.difficulty / self.trial
#         # self.img_direction = ImageDirection[job_params["img_direction"]]
#         self.img_direction: ImageDirection = self._discern_img_direction(
#             job_params["img_direction"])

#         # TODO: Make this an enum in the projection based flow stuff
#         self.method: str = job_params["method"]
#         self.using_ground_truth_flow: bool = job_params["using_ground_truth_flow"]

#     def cache_job(self) -> Path:
#         out_path = CACHED_JOBS_DIR / f"{hash(self)}.yml"
#         with out_path.open('w') as f:
#             yaml.dump(self._job_params, f)
#         return out_path

#     def _discern_img_direction(self, img_direction_str: str) -> ImageDirection:
#         if img_direction_str == "left":
#             return ImageDirection.LEFT
#         elif img_direction_str == "right":
#             return ImageDirection.RIGHT
#         else:
#             raise ValueError(
#                 f"Invalid image direction: {img_direction_str}, expected 'left' or 'right'")

#     def __eq__(self, other):
#         if not isinstance(other, Job):
#             return False
#         return all(self.scenario == other.scenario, self.difficulty == other.difficulty,
#                    self.trial == other.trial, self.img_direction == other.img_direction,
#                    self.method == other.method)


class Config:

    def __init__(self, job_dicts: Optional[List[Dict]] = None, job_cfg_path: Path = JOB_CFG_PATH):
        if job_dicts is not None:
            self._job_params: List[Dict] = job_dicts
        with job_cfg_path.open('r') as f:
            self._job_params: List[Dict] = yaml.safe_load(f)["jobs"]

        self.jobs = [Job.from_cfg_dict(job) for job in self._job_params]

    @staticmethod
    def list_cache_paths():
        entries = []
        for job in CACHED_JOBS_DIR.iterdir():
            entries.append(job)
        return entries

    @staticmethod
    def read_cache():
        # Cache is now a directory that has hashed job configs as files.
        job_dicts = []
        for job in Config.list_cache_paths():
            # with job.open('r') as f:
            #     job_dicts.append(yaml.safe_load(f))
            with job.open('rb') as f:
                job_dicts.append(pkl.load(f))
        return Config(job_dicts=job_dicts)

    def update_cache(self):
        # try:
        #     cfg_cache = Config.read_cache()
        # except (FileNotFoundError, KeyError):
        #     # Key error is for when the new job config contains a new parameter the old one didn't.
        #     cfg_cache = Config([])

        # # These should probably be sets instead of lists.
        # jobs_missing = self.find_jobs_not_in_other_config(cfg_cache)
        # new_paths = []
        # for job in jobs_missing:
        #     new_path = job.cache_job()
        #     new_paths.append(new_path)

        # return new_paths

        for job in self.jobs:
            job.cache_job()

    # def __eq__(self, value: object) -> bool:
    #     if not isinstance(value, Config):
    #         return False

    #     return all([job in value.jobs for job in self.jobs])

    def find_jobs_not_in_other_config(self, other: 'Config'):
        return [job for job in self.jobs if job not in other.jobs]

    def is_job_present(self, job: Job):
        return job in self.jobs


# def task_cache_cfg():

#     def cache_cfg():
#         cfg = Config()
#         cfg.cache_cfg()

#     return {
#         "actions": [(cache_cfg, [])],
#         "file_dep": [JOB_CFG_PATH],
#         "targets": [CACHED_JOBS_PATH],
#     }


def idx_target_file_from_cfg(cfg: Config) -> Path:
    return ROOT_DIR / f"max_image_idx_{cfg.img_direction}.txt"


def frame_idx_to_str(frame_idx: int) -> str:
    return f"{frame_idx:06d}"


def get_mask_output_path(cfg: Config, frame_idx_t: int) -> Path:
    f_t = frame_idx_to_str(frame_idx_t)
    name = f"{f_t}_{cfg.img_direction.value}_mask.png"

    gt_str = "gt" if cfg.using_ground_truth_flow else "pred"
    direction_dir = cfg.trial_path / f"dynamic_obj_masks_{cfg.img_direction.value}"
    out_dir = direction_dir / f"{cfg.method}_method" / gt_str

    return out_dir / name


def save_mask(mask: np.ndarray, out_path: Path):
    mask = mask.astype(np.uint8) * 255
    mask = Image.fromarray(mask)
    mask.save(out_path)


def get_max_img_idx(trial_path: Path, img_direction: ImageDirection) -> int:
    img_gt_dir = trial_path / f"image_{img_direction.value}"
    img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
    max_idx = max(img_idxs)
    return max_idx


def process_single_frame_pair(cfg: Config, frame_idx: int, poses: np.ndarray):
    i = frame_idx - 1
    j = frame_idx

    out_path = get_mask_output_path(cfg, j)

    (rgb_2, depth_2) = read_rgbd(cfg.trial_path, cfg.img_direction, j)
    if frame_idx == 0:
        # No previous frame to compare to, so we indicate that nothing is dynamic.
        save_mask(np.zeros_like(depth_2), out_path)
        return

    (rgb_1, depth_1) = read_rgbd(cfg.trial_path, cfg.img_direction, i)

    poses = read_pose(cfg.trial_path, cfg.img_direction)
    pose_1 = get_transform_matrix_from_pose_array(poses[i, :])
    pose_2 = get_transform_matrix_from_pose_array(poses[j, :])

    flow = OpticalFlow(None)
    flow.load_model(RAFT_MODEL_PATH, Args())

    mask = projection_based_motion_segmentation(rgb_1, depth_1, rgb_2, depth_2, pose_1, pose_2,
                                                flow)

    save_mask(mask, out_path)


def spawn_job(job: Job):
    job_path = job.get_cache_path()
    max_idx = get_max_img_idx(job.trial_path, job.img_direction)

    # Get a dummy mask output path to make that directory.
    dummy_out_path = get_mask_output_path(job, 0)
    dummy_out_path.parent.mkdir(exist_ok=True, parents=True)

    def define_single_mask_tasks():
        poses = read_pose(job.trial_path, job.img_direction)

        for idx_t in range(max_idx + 1):
            # this doesn't do the actual processing, it just defines the tasks.
            f_t = frame_idx_to_str(idx_t)
            targets = get_mask_output_path(job, idx_t)
            # print("targets: ", targets)

            yield {
                "name": f"process_mask_job_{job.name}_{f_t}",
                "actions": [(process_single_frame_pair, [job, idx_t, poses])],
                "file_dep": [job_path],
                "targets": [targets],
            }

    yield define_single_mask_tasks()


# def task_process_masks():

# def task_run_all():
#     yield {
#         "actions": None,
#         "task_dep": ["cache_cfg", "process_masks"],
#     }


def task_spawn_jobs():
    # cfg_cache = Config.read_cache()
    cfg = Config()

    cfg.update_cache()

    for job in Config.read_cache().jobs:
        yield spawn_job(job)


# def task_zip_masks():

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
