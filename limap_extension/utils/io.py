from pathlib import Path
import sys

import numpy as np
from PIL import Image

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(FILE_DIR.as_posix())
from path_fixer import allow_limap_imports

allow_limap_imports()

from limap_extension.constants import ImageType, ImageDirection


def read_img(trial_path: Path, img_type: ImageType, img_direction: ImageDirection,
             frame: int) -> np.array:
    """Reads the image at the given path."""
    img_dir = trial_path / f"{img_type.value}_{img_direction.value}"
    img_name = f"{frame:06d}_{img_direction.value}"

    if img_type == ImageType.DEPTH:
        img_name += f"_{img_type.value}.npy"
        img_path = img_dir / img_name
        img = np.load(img_path)
    elif img_type == ImageType.IMAGE:
        img_name += ".png"
        img_path = img_dir / img_name
        img = Image.open(img_path)
        img = np.asarray(img)
    else:
        raise ValueError("Invalid image type")

    return img


def read_rgbd(trial_path: Path, img_direction: ImageDirection, frame: int) -> np.array:
    """Reads the RGBD image at the given path."""
    img = read_img(trial_path, ImageType.IMAGE, img_direction, frame)
    depth = read_img(trial_path, ImageType.DEPTH, img_direction, frame)
    return img, depth


def read_pose(trial_path: Path, img_direction: ImageDirection) -> np.ndarray:
    """Reads the pose at the given path."""
    pose_path = trial_path / f"pose_{img_direction.value}.txt"
    pose = np.loadtxt(pose_path)
    return pose
