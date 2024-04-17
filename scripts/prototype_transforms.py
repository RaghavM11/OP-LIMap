import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2

REPO_DIR = Path(__file__).resolve().parents[1]
# print(REPO_DIR)
sys.path.append(REPO_DIR.as_posix())

from limap_extension.optical_flow import motion_segmentation, Args, OpticalFlow
from limap_extension.img_cloud_transforms import (reproject_img, index_img_with_uv_coords,
                                                  get_uv_coords, imgs_to_clouds_np, tform_coords,
                                                  cam2ned_single_pose, ned2cam_single_pose)
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array
from limap_extension.bounding_box import BoundingBox

# from limap_extension.visualization.rerun.figure_factory import FigureFactory

# TRIAL_DIR = REPO_DIR / "datasets" / "ocean" / "Hard" / "P006"
TRIAL_DIR = REPO_DIR / "datasets" / "carwelding" / "easy" / "P007"
FRAME_1 = 100
FRAME_2 = 110
RAFT_PATH = REPO_DIR / "raft-sintel.pth"

# Load in the data.
frame_str = f"{FRAME_1:06d}"
rgb_1 = np.array(Image.open(TRIAL_DIR / "image_left" / f"{frame_str}_left.png"))
depth_1 = np.load(TRIAL_DIR / "depth_left" / f"{frame_str}_left_depth.npy")

frame_str = f"{FRAME_2:06d}"
rgb_2 = np.array(Image.open(TRIAL_DIR / "image_left" / f"{frame_str}_left.png"))
depth_2 = np.load(TRIAL_DIR / "depth_left" / f"{frame_str}_left_depth.npy")

cam_poses = np.loadtxt(TRIAL_DIR / "pose_left.txt")

# Opting to crop the depth images before input to reprojection
depth_max = 50
depth_1 = np.clip(depth_1, None, depth_max)
depth_2 = np.clip(depth_2, None, depth_max)


def display_img_pair(rgb, depth, img_slice: BoundingBox = None):
    if img_slice is not None:
        rgb = img_slice.crop_img(rgb)
        depth = img_slice.crop_img(depth)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)

    # depth = np.clip(depth, 0, 10)
    im1 = ax[1].imshow(depth)
    fig.colorbar(im1, ax=ax[1])


# crop_box = BoundingBox(200, 350, 125, 275)
# crop_box = BoundingBox(0, 100, 350, -1)
# crop_box = BoundingBox(325, 450, 180, 220)
# crop_box = BoundingBox(0, -1, 0, -1)
# display_img_pair(rgb_1, depth_1, crop_box)
# display_img_pair(rgb_2, depth_2, crop_box)

flow = OpticalFlow(None)
flow.load_model(RAFT_PATH, Args())

cp1 = cam_poses[FRAME_1, :]
cp2 = cam_poses[FRAME_2, :]

pose_1 = get_transform_matrix_from_pose_array(cp1)
pose_2 = get_transform_matrix_from_pose_array(cp2)

# print("Pose 1:", pose_1)
# print("Pose 2:", pose_2)

tx_true = pose_2[:3, 3] - pose_1[:3, 3]
print("True delta position:", tx_true)

# dpos = np.linalg.inv(pose_1) @ pose_2
# dpos = pose_1 @ np.linalg.inv(pose_2)
# dpos = np.linalg.inv(np.linalg.inv(pose_1) @ pose_2)
# dpos = np.linalg.inv(np.linalg.inv(pose_2) @ pose_1)
# tx_test = dpos[:3, 3]
# print("Pose delta:", dpos)
# print("Test delta position:", tx_test)

# print(us.max())
# print(us[:10])
# print(us_out[:10])

import rerun as rr
from limap_extension.constants import CAM_INTRINSIC

cloud_1_frame_1, _ = imgs_to_clouds_np(rgb_1, depth_1, CAM_INTRINSIC)
cloud_2_frame_2, _ = imgs_to_clouds_np(rgb_2, depth_2, CAM_INTRINSIC)

rr.init("tester", spawn=False)
rr.serve()
rr.set_time_seconds("stable", 0)

import time

time.sleep(7)

# rr_kwargs = cloud_1.form_rerun_kwargs()
# rr.log("cloud_1", rr.Points3D(**rr_kwargs))


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    R = pose[:3, :3]
    t = pose[:3, 3]
    inv_R = R.T
    # inv_R = np.eye(3)
    inv_t = -inv_R @ t
    # inv_t = -t
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = inv_R
    inv_pose[:3, 3] = inv_t
    return inv_pose


cloud_1_frame_0 = cloud_1_frame_1.copy()
# tform = np.linalg.inv(pose_1)
# tform_world_to_cam = inverse_pose(pose_1)
# tform_world_to_ned = cam2ned(tform)
# tform = tform_world_to_cam
# tform = np.eye(4)
tform = pose_1
# tform = np.eye(4)
# cloud_1_frame_0.apply_extrinsic(tform)
cloud_1_frame_0.xyz = tform_coords(tform, cloud_1_frame_1.xyz)

cloud_2_frame_0 = cloud_2_frame_2.copy()
# tform = inverse_pose(pose_2)
# tform = np.linalg.inv(pose_2)
tform = pose_2
# tform = np.eye(4)
cloud_2_frame_0.xyz = tform_coords(tform, cloud_2_frame_2.xyz)
# cloud_2_frame_0.apply_extrinsic(tform)

rr_kwargs = cloud_1_frame_0.form_rerun_kwargs()
rr.log("cloud_1_frame_0", rr.Points3D(**rr_kwargs))

rr_kwargs = cloud_2_frame_0.form_rerun_kwargs()
rr.log("cloud_2_frame_0", rr.Points3D(**rr_kwargs))
