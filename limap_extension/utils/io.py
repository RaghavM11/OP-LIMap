from pathlib import Path
import sys
from typing import Tuple, List

import numpy as np
from PIL import Image

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(FILE_DIR.as_posix())
from path_fixer import allow_limap_imports

allow_limap_imports()

from limap_extension.constants import ImageType, ImageDirection


def get_img_path(trial_path: Path, img_type: ImageType, img_direction: ImageDirection, frame: int):
    img_dir = trial_path / f"{img_type.value}_{img_direction.value}"
    img_name = f"{img_direction.value}.{img_type.value}"

    img_name = f"{frame:06d}_{img_direction.value}"

    if img_type == ImageType.DEPTH:
        img_name += f"_{img_type.value}.npy"
        img_path = img_dir / img_name
    elif img_type == ImageType.IMAGE:
        img_name += ".png"
        img_path = img_dir / img_name
    else:
        raise ValueError("Invalid image type")

    return img_path


def read_img(trial_path: Path, img_type: ImageType, img_direction: ImageDirection,
             frame: int) -> np.array:
    """Reads the image at the given path."""
    img_path = get_img_path(trial_path, img_type, img_direction, frame)

    if img_type == ImageType.DEPTH:
        img = np.load(img_path)
    elif img_type == ImageType.IMAGE:
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


def read_all_rgbd(
    trial_path: Path, img_direction: ImageDirection
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
    # Going to read in poses as a hack to figure out how many images we have.
    poses = read_pose(trial_path, img_direction)
    num_images = poses.shape[0] + 1
    rgbs = []
    rgb_paths = []
    depths = []
    depth_paths = []
    for img_idx in range(num_images):
        rgb, depth = read_rgbd(trial_path, img_direction, img_idx)
        rgbs.append(rgb)
        depths.append(depth)

        rgb_path = get_img_path(trial_path, ImageType.IMAGE, img_direction, img_idx).as_posix()
        rgb_paths.append(rgb_path)
        depth_path = get_img_path(trial_path, ImageType.DEPTH, img_direction, img_idx).as_posix()
        depth_paths.append(depth_path)
    return rgbs, rgb_paths, depths, depth_paths


def read_pose(trial_path: Path, img_direction: ImageDirection) -> np.ndarray:
    """Reads the pose at the given path."""
    pose_path = trial_path / f"pose_{img_direction.value}.txt"
    pose = np.loadtxt(pose_path)
    return pose
