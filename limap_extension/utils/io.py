from pathlib import Path
import sys
from typing import Tuple, List

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import binary_dilation

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(FILE_DIR.as_posix())
from path_fixer import allow_limap_imports

allow_limap_imports()

from limap_extension.constants import ImageType, ImageDirection


def save_mask(mask: np.ndarray, out_path: Path):
    mask = mask.astype(np.uint8) * 255
    mask = Image.fromarray(mask)
    mask.save(out_path)


def save_rgb(rgb: np.ndarray, out_path: Path):
    if rgb.dtype != np.uint8:
        raise ValueError("The input image must be of type np.uint8.")
    rgb = Image.fromarray(rgb)
    rgb.save(out_path)


def scale_flow_viz(flow_viz: np.ndarray, power: float = 3.0) -> np.ndarray:
    if flow_viz.max() > 1:
        flow_viz = flow_viz.astype(float) / 255
    flow_viz = flow_viz**power
    return np.round(flow_viz * 255).astype(np.uint8)


def flow_xyz_to_rgb(flow: 'OpticalFlow', flow_xyz: np.ndarray) -> np.ndarray:
    flow_mag = np.linalg.norm(flow_xyz, axis=-1)
    flow_planar = flow_xyz[:, :, :2]
    flow_planar_mag = np.linalg.norm(flow_planar, axis=-1)
    flow_planar = flow_planar / flow_planar_mag[..., None]
    flow_planar = flow_planar * flow_mag[..., None]
    flow_reproj_rgb = flow.visualize(torch.from_numpy(flow_planar).permute(2, 0, 1).unsqueeze(0))


def flow_mag_angle_to_rgb(flow_mag: np.ndarray, flow_angle: np.ndarray) -> np.ndarray:
    """Converts flow magnitude and angle to an RGB image."""
    mask = np.zeros((*flow_mag.shape, 3), dtype=np.float32)
    print("Max flow magnitude: ", np.max(flow_mag))
    print("Min flow magnitude: ", np.min(flow_mag))
    print("Max flow angle: ", np.max(flow_angle))
    print("Min flow angle: ", np.min(flow_angle))

    mask[..., 0] = 0  #flow_angle
    mask[..., 1] = 255
    mask[..., 2] = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB_FULL).astype(float)
    mask_r = np.mean(flow_rgb[:, :, 0])
    mask_g = np.mean(flow_rgb[:, :, 1])
    mask_b = np.mean(flow_rgb[:, :, 2])
    cov_mask_r = np.std(flow_rgb[:, :, 0])
    cov_mask_g = np.std(flow_rgb[:, :, 1])
    cov_mask_b = np.std(flow_rgb[:, :, 2])
    mask_r = (flow_rgb[:, :, 0] - mask_r) / cov_mask_r
    mask_g = (flow_rgb[:, :, 1] - mask_g) / cov_mask_g
    mask_b = (flow_rgb[:, :, 2] - mask_b) / cov_mask_b
    return np.clip(flow_rgb * 255, 0, 255).astype(np.uint8)


def flow_2d_to_rgb_and_mask(flow_up: np.ndarray) -> np.ndarray:
    """Converts a 2D flow image to an RGB image."""
    mask = np.zeros((*flow_up.shape[:-1], 3), dtype=np.uint8)
    mag, angle = cv2.cartToPolar(flow_up[..., 0], flow_up[..., 1], angleInDegrees=True)

    # mask[:, :, 0] = angle
    # mask[..., 1] = 255
    # mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # print(np.mean(mag), np.max(mag), np.min(mag))
    # flow_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB).astype(float)
    # mean = np.mean(flow_rgb, axis=(0, 1), keepdims=True)
    # std_deviation = np.std(flow_rgb, axis=(0, 1), keepdims=True)

    # # mask_grey = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2GRAY)
    # flow_rgb = np.round(((flow_rgb - mean) / std_deviation) * 255).astype(np.uint8)

    # mask_grey = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2GRAY)
    # mask_grey = (mask_grey - mean) / std_deviation

    # # do binary dilation on the grey_mask
    # # mask_grey = binary_dilation(mask_grey, iterations=5)
    # mask_grey = np.clip(mask_grey, 0, None)
    # mask_grey[mask_grey > 0.5] = 1
    # mask_grey = binary_dilation(mask_grey, iterations=5)

    # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
    mask_rgb = flow_mag_angle_to_rgb(mag, angle)
    mask_grey = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    mean = np.mean(mask_grey)
    # # print(mean)
    std_deviation = np.std(mask_grey)
    # # print(std_deviation)
    mask_grey = (mask_grey - mean) / std_deviation
    # # mask_grey = cv2.normalize(mask_grey, None, 0, 1, cv2.NORM_MINMAX)
    mask_grey = np.where(mask_grey < 0, 0, mask_grey)
    mask_grey = np.where(mask_grey > 0.5, 1, 0)
    # # print(np.mean(mask_grey[np.where(mask_grey > 0)]))

    print(
        "This normalization means that the RGB conversion will be clipped to within one standard deviation."
    )
    # mask_rgb = np.stack([mask_r, mask_g, mask_b], axis=2)
    # mask_rgb =
    mask_rgb = np.round(mask_rgb * 255).astype(np.uint8)
    # mask_grey = cv2.cvtColor(mask_rgb*255, cv2.COLOR_RGB2GRAY)
    # mask_grey =  mask_grey * 255

    # do binary dilation on the grey_mask
    mask_grey = binary_dilation(mask_grey, iterations=5)

    return mask_rgb, mask_grey


def get_img_path(trial_path: Path, img_type: ImageType, img_direction: ImageDirection, frame: int):
    if isinstance(trial_path, str):
        trial_path = Path(trial_path)
    img_dir = trial_path / f"{img_type.value}_{img_direction.value}"
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
    if isinstance(trial_path, str):
        trial_path = Path(trial_path)
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
    if isinstance(trial_path, str):
        trial_path = Path(trial_path)
    img = read_img(trial_path, ImageType.IMAGE, img_direction, frame)
    depth = read_img(trial_path, ImageType.DEPTH, img_direction, frame)
    return img, depth


def read_all_rgbd(
    trial_path: Path, img_direction: ImageDirection
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
    # Going to read in poses as a hack to figure out how many images we have.
    if isinstance(trial_path, str):
        trial_path = Path(trial_path)
    poses = read_pose(trial_path, img_direction)
    num_images = poses.shape[0]
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
    if isinstance(trial_path, str):
        trial_path = Path(trial_path)
    pose_path = trial_path / f"pose_{img_direction.value}.txt"
    pose = np.loadtxt(pose_path)
    return pose
