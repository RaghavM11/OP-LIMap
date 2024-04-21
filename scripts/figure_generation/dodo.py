from pathlib import Path
import sys

import yaml

FILE_DIR = Path(__file__).resolve().parent
REPO_DIR = FILE_DIR.parents[1]
sys.path.append(REPO_DIR.as_posix())
SCRIPTS_DIR = REPO_DIR / "scripts"
sys.path.append(SCRIPTS_DIR.as_posix())

from limap_extension.constants import ImageDirection
from scripts.figure_generation.figure_optical_flow import FigureGenerator

JOB_FILE = FILE_DIR / "job.yml"
OUTPUT_DIR = FILE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_job():
    with open(JOB_FILE, "r") as f:
        job = yaml.safe_load(f)
    job["img_direction"] = (ImageDirection.LEFT
                            if job["img_direction"] == "left" else ImageDirection.RIGHT)
    return job


def task_generate_flow_figure():
    job = load_job()
    fg = FigureGenerator(**job, output_dir=OUTPUT_DIR)
    all_targets = fg.get_target_files()
    fig_name = fg.get_figure_path().stem

    # Generate figures for each scenario, difficulty, trial, image direction, and frame index.
    yield {
        "name": f"generate_{fig_name}",
        "actions": [(fg.generate_figure, [])],
        "file_dep": [JOB_FILE],
        "targets": all_targets,
    }
