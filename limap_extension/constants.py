from enum import Enum
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parent
DATASET_DIR = REPO_DIR / "datasets"


class ImageType(Enum):
    """Enum for the different types of images."""
    DEPTH = "depth"
    IMAGE = "image"


class ImageDirection(Enum):
    """Enum for the directions images were taken in (left/right)"""
    LEFT = "left"
    RIGHT = "right"


# yapf: disable
CAM_INTRINSIC = np.array((
    (320,   0, 320),
    (  0, 320, 240),
    (  0,   0,   1)), dtype=float)

H_CAM_TO_NED = np.array((
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (1, 0, 0, 0),
    (0, 0, 0, 1)
), dtype=np.float32)

H_NED_TO_CAM = np.array((
    (0, 0, 1, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 0, 1)
), dtype=np.float32)
# yapf: enable
