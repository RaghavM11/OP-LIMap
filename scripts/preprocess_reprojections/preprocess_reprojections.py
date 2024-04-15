"""Pre-processes and saves the reprojection data for the given dataset."""
from pathlib import Path
import sys
from enum import Enum
import pickle as pkl

from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# print("Repo root:", REPO_ROOT)
sys.path.append(REPO_ROOT.as_posix())

from limap_extension.img_cloud_transforms import reproject_img
from limap_extension.bounding_box import BoundingBox

DATASET_DIR = REPO_ROOT / "datasets"

SCENARIO = "ocean"
DIFFICULTY = "Hard"
TRIAL = "P006"


class ImageType(Enum):
    """Enum for the different types of images."""
    DEPTH = "depth"
    IMAGE = "image"


class ImageDirection(Enum):
    """Enum for the directions images were taken in (left/right)"""
    LEFT = "left"
    RIGHT = "right"


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


def get_img_bbox_paths(dataset_path: Path, img_direction: ImageDirection, frame_i: int,
                       frame_j: int):
    img_dir = dataset_path / f"reproj_{img_direction.value}"
    img_dir.mkdir(exist_ok=True)

    frame_indicator = f"{frame_i:06d}_{frame_j:06d}"
    img_name = f"{frame_indicator}_reproj_{img_direction.value}.png"
    bbox_name = f"{frame_indicator}_bbox_{img_direction.value}.pkl"

    img_path = img_dir / img_name
    bbox_path = img_dir / bbox_name

    return img_path, bbox_path


def save_reprojection(dataset_path: Path, img_reproj: np.ndarray, valid_bbox: BoundingBox,
                      img_direction: ImageDirection, frame_i: int, frame_j: int):

    img_path, bbox_path = get_img_bbox_paths(dataset_path, img_direction, frame_i, frame_j)

    # np.save(img_path, img_reproj)

    img_reproj_pil = Image.fromarray(img_reproj)
    img_reproj_pil.save(img_path)

    print("Saved reprojection to:", img_path)

    with open(bbox_path, "wb") as f:
        pkl.dump(valid_bbox, f)

    print("Saved bounding box to:", bbox_path)

    return img_path, bbox_path


def main():
    dataset_path = DATASET_DIR / SCENARIO / DIFFICULTY / TRIAL
    print("Dataset path:", dataset_path)

    img_type = ImageType.DEPTH
    img_direction = ImageDirection.LEFT
    poses = read_pose(dataset_path, img_direction)

    (rgb_1, depth_1) = read_rgbd(dataset_path, img_direction, 0)
    pose_1 = poses[0, :]
    # for i in range(1, poses.shape[0]):
    for i in range(1, 3):
        (rgb_2, depth_2) = read_img(dataset_path, img_type, img_direction, i)
        pose_2 = poses[i, :]

        img_1_in_frame_2, valid_bbox = reproject_img(rgb_1, depth_1, pose_1, pose_2)

        save_reprojection(dataset_path, img_1_in_frame_2, valid_bbox, img_direction, i - 1, i)

        rgb_1 = rgb_2
        depth_1 = depth_2
        pose_1 = pose_2


if __name__ == "__main__":
    main()
