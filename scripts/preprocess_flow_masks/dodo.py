from pathlib import Path
import sys
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from copy import deepcopy
import hashlib

import numpy as np
from PIL import Image
import yaml

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parents[1]
sys.path.append(REPO_DIR.as_posix())

from limap_extension.utils.io import read_rgbd, read_pose, save_mask
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


def frame_idx_to_str(frame_idx: int) -> str:
    return f"{frame_idx:06d}"


@dataclass(frozen=True, eq=True)
class Job:
    scenario: str
    difficulty: str
    trial: str
    img_direction: ImageDirection
    method: str
    using_ground_truth_flow: bool
    trial_path: Path

    @staticmethod
    def from_cfg_dict(cfg: Dict) -> 'Job':
        kwargs = {}
        if "trial_path" not in cfg:
            kwargs["trial_path"] = DATASET_DIR / cfg["scenario"] / cfg["difficulty"] / cfg["trial"]

        # This modifies the original config dict but I won't worry about that now.
        cfg["img_direction"] = (ImageDirection.LEFT
                                if cfg["img_direction"] == "left" else ImageDirection.RIGHT)
        return Job(**cfg, **kwargs)  #, trial_path=trial_path)

    def get_hash(self) -> str:
        serial = self.serialize()
        # I think SHA-256 is hilariously overkill but it's fine.
        serial_hash = hashlib.sha256(serial.encode()).hexdigest()

        # Truncate the hash to 8 characters.
        return serial_hash[:8]

    def get_cache_path(self) -> Path:
        return CACHED_JOBS_DIR / f"{self.get_hash()}.yml"

    def cache_job(self) -> Path:
        out_path = self.get_cache_path()
        if out_path.exists():
            return out_path

        with out_path.open('w') as f:
            yaml.dump(self.to_yamlable_dict(), f)

        return out_path

    def to_yamlable_dict(self) -> Dict[str, Any]:
        d = deepcopy(asdict(self))

        # Convert non-yaml-able members to seralized representation.
        d["img_direction"] = d["img_direction"].value
        # I think Path objects are yaml-able, but just to be safe.
        d["trial_path"] = d["trial_path"].as_posix()
        return d

    def serialize(self) -> str:
        # Sort to ensure that the serialized representation is deterministic.
        return str(dict(sorted(self.to_yamlable_dict().items())))

    def get_mask_output_path(self, frame_idx_t: int) -> Path:
        f_t = frame_idx_to_str(frame_idx_t)
        name = f"{f_t}_{self.img_direction.value}_mask.png"

        gt_str = "gt" if self.using_ground_truth_flow else "pred"
        direction_dir = self.trial_path / f"dynamic_obj_masks_{self.img_direction.value}"
        out_dir = direction_dir / f"{self.method}_method" / gt_str

        return out_dir / name

    def get_max_img_idx(self) -> int:
        img_gt_dir = self.trial_path / f"image_{self.img_direction.value}"
        img_idxs = [int(img_path.stem.split("_")[0]) for img_path in img_gt_dir.iterdir()]
        max_idx = max(img_idxs)
        return max_idx

    def get_all_mask_targets(self) -> List[Path]:
        max_idx = self.get_max_img_idx(self.trial_path, self.img_direction)
        return [self.get_mask_output_path(i) for i in range(max_idx + 1)]


class JobList:

    def __init__(self, job_dicts: Optional[List[Dict]] = None, job_cfg_path: Path = JOB_CFG_PATH):
        if job_dicts is not None:
            self._job_params: List[Dict] = job_dicts
        else:
            with job_cfg_path.open('r') as f:
                self._job_params: List[Dict] = yaml.safe_load(f)["jobs"]

        self.jobs = [Job.from_cfg_dict(job) for job in self._job_params]

    @staticmethod
    def list_cache_paths() -> List[Path]:
        entries = []
        for job_path in CACHED_JOBS_DIR.iterdir():
            if job_path.is_file() and job_path.suffix == ".yml":
                entries.append(job_path)
        return entries

    @staticmethod
    def from_cache():
        # Cache is now a directory that has hashed job configs as files.
        job_dicts = []
        for job in JobList.list_cache_paths():
            with job.open('r') as f:
                job_dicts.append(yaml.safe_load(f))
        return JobList(job_dicts=job_dicts)

    @staticmethod
    def clean_cache():
        for job in JobList.list_cache_paths():
            job.unlink()

    def update_cache(self):
        # If cache loading is slow, could be worth it to turn self into a dictionary of hash-job
        # pairs and compare that with the list of cache files.
        cache = JobList.from_cache()
        new_jobs = set(self.jobs) - set(cache.jobs)
        for job in new_jobs:
            print("Caching job:", job.get_hash())
            job.cache_job()


def process_single_frame_pair(job: Job, frame_idx: int, poses: np.ndarray):
    i = frame_idx - 1
    j = frame_idx

    out_path = job.get_mask_output_path(j)

    (rgb_2, depth_2) = read_rgbd(job.trial_path, job.img_direction, j)
    if frame_idx == 0:
        # No previous frame to compare to, so we indicate that nothing is dynamic.
        save_mask(np.zeros_like(depth_2), out_path)
        return

    (rgb_1, depth_1) = read_rgbd(job.trial_path, job.img_direction, i)

    poses = read_pose(job.trial_path, job.img_direction)
    pose_1 = get_transform_matrix_from_pose_array(poses[i, :])
    pose_2 = get_transform_matrix_from_pose_array(poses[j, :])

    flow = OpticalFlow(None)
    flow.load_model(RAFT_MODEL_PATH, Args())

    mask = projection_based_motion_segmentation(rgb_1, depth_1, rgb_2, depth_2, pose_1, pose_2,
                                                flow)

    save_mask(mask, out_path)


def spawn_job(job: Job):
    job_path = job.get_cache_path()
    max_idx = job.get_max_img_idx()

    # Get a dummy mask output path to make that directory. Probably wouldn't impact performance to
    # not do this and just call mkdir in the loop but I sure love premature optimization.
    dummy_out_path = job.get_mask_output_path(0)
    dummy_out_path.parent.mkdir(exist_ok=True, parents=True)

    def define_single_mask_tasks():
        poses = read_pose(job.trial_path, job.img_direction)

        for idx_t in range(max_idx + 1):
            # this doesn't do the actual processing, it just defines the tasks.
            f_t = frame_idx_to_str(idx_t)
            targets = job.get_mask_output_path(idx_t)

            yield {
                "name": f"process_mask_job_{job.get_hash()}_{f_t}",
                "actions": [(process_single_frame_pair, [job, idx_t, poses])],
                "file_dep": [job_path],
                "targets": [targets],
            }

    yield define_single_mask_tasks()


def task_preprocess_flow_masks():
    job_list = JobList()

    job_list.update_cache()

    for job in job_list.jobs:
        yield spawn_job(job)


def task_zip_trials():

    def get_zip_target(job: Job):
        return DATASET_DIR / f"{job.scenario}_{job.difficulty}_{job.trial}_with_flow_masks.zip"

    def zip_trial(job: Job):
        import zipfile

        out_path = get_zip_target(job)
        trial_dir = DATASET_DIR / job.scenario / job.difficulty / job.trial
        paths_to_zip = trial_dir.glob("./**/*")

        with zipfile.ZipFile(out_path.as_posix(), 'w') as z:
            # Want to write the zip file in a way that is easy to extract, so chop off everything
            # prior to the dataset. That way, dataset structure if consistent across machines.
            for mask_path in paths_to_zip:
                arc_name = mask_path.relative_to(DATASET_DIR)
                z.write(mask_path, arc_name)

    for job in JobList().jobs:
        file_dep = [job.get_cache_path()]
        zip_target = get_zip_target(job)
        yield {
            "basename": f"zip_trials_{job.get_hash()}",
            "actions": [(zip_trial, [job])],
            "file_dep": file_dep,
            "targets": [zip_target],
        }
