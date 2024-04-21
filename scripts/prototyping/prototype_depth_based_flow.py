import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2

REPO_DIR = Path(__file__).resolve().parents[2]
# print(REPO_DIR)
sys.path.append(REPO_DIR.as_posix())

from limap_extension.optical_flow import motion_segmentation, Args, OpticalFlow
from limap_extension.img_cloud_transforms import reproject_img, uvz_ocv_to_xyz_ned, xyz_ned_to_uvz_ocv, index_img_with_uv_coords, get_uv_coords, imgs_to_clouds_np
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array
from limap_extension.bounding_box import BoundingBox

# from limap_extension.visualization.rerun.figure_factory import FigureFactory

# TRIAL_DIR = REPO_DIR / "datasets" / "ocean" / "Hard" / "P006"
TRIAL_DIR = REPO_DIR / "datasets" / "carwelding" / "easy" / "P007"
# FRAME_1 = 566
# FRAME_2 = 567
FRAME_1 = 127
FRAME_2 = 128
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
dpos = np.linalg.inv(np.linalg.inv(pose_2) @ pose_1)
tx_test = dpos[:3, 3]
# print("Pose delta:", dpos)
print("Test delta position:", tx_test)

# print(us.max())
# print(us[:10])
# print(us_out[:10])

import rerun as rr
from limap_extension.constants import CAM_INTRINSIC

cloud_1, _ = imgs_to_clouds_np(rgb_1, depth_1, CAM_INTRINSIC)

rr.init("tester", spawn=False)
rr.serve()
rr.set_time_seconds("stable", 0)

import time

time.sleep(7)

# rr_kwargs = cloud_1.form_rerun_kwargs()
# rr.log("cloud_1", rr.Points3D(**rr_kwargs))

# Reproject the image at time t to the image frame at time t+1
img_1_in_frame_2, depth_1_in_frame_2, mask_valid_projection, valid_bbox = reproject_img(
    rgb_1, depth_1, pose_1, pose_2)

# display_img_pair(img_1_in_frame_2, depth_1_in_frame_2)

valid_bbox = BoundingBox(0, -1, 0, -1)

print(valid_bbox)

rgb_1_cropped = valid_bbox.crop_img(img_1_in_frame_2)
depth_1_cropped = valid_bbox.crop_img(depth_1_in_frame_2)
rgb_2_cropped = valid_bbox.crop_img(rgb_2)
depth_2_cropped = valid_bbox.crop_img(depth_2)

# display_img_pair(valid_bbox.crop_img(rgb_1), valid_bbox.crop_img(depth_1), crop_box)
# display_img_pair(rgb_1_cropped, depth_1_cropped, crop_box)

# # fig, ax = plt.subplots(1, 2)
# # ax[0].imshow(rgb_2_cropped)
# # ax[1].imshow(depth_2_cropped)
# display_img_pair(rgb_2_cropped, depth_2_cropped, crop_box)

# rgb_disparity = np.linalg.norm(rgb_1_cropped.astype(float) - rgb_2_cropped.astype(float), axis=-1)

# # mean_disparity = np.mean(rgb_disparity)
# # thresh_min = mean_disparity + np.std(rgb_disparity)
# thresh_min = 0
# rgb_disparity = np.clip(rgb_disparity, thresh_min, None)

# # plt.figure()
# # plt.imshow(rgb_disparity)
# # plt.title("RGB Disparity")

# depth_disparity = np.abs(depth_1_cropped - depth_2_cropped)
# print("Min depth disparity:", np.min(depth_disparity))
# # mean_disparity = np.mean(depth_disparity)
# # thresh_min = mean_disparity + np.std(depth_disparity)
# # depth_disparity = np.clip(depth_disparity, thresh_min, None)
# # depth_disparity = np.clip(depth_disparity, 0, 0.2)
# # plt.figure()
# # plt.imshow(depth_disparity)
# # plt.title("Depth Disparity")
# # plt.colorbar()

# depth_disparity = np.clip(depth_disparity, 0, 0.5)
# display_img_pair(rgb_disparity, depth_disparity, crop_box)

flow_low, flow_up = flow.infer_flow(rgb_1_cropped, rgb_2_cropped)
# flow_low, flow_up = flow.infer_flow(rgb_2_cropped, rgb_1_cropped)
# flow_low, flow_up = flow.infer_flow(rgb_1, rgb_2)

print("Cropped shape:", rgb_1_cropped.shape)
print("Flow up shape:", flow_up.shape)

# print(flow_up, flow_low)
flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()

# flow_up.shape

# plt.figure()
# plt.imshow(crop_box.crop_img(flow_up[..., 0]))
# plt.colorbar()
# plt.title("Flow u")

# plt.figure()
# plt.imshow(crop_box.crop_img(flow_up[..., 1]))
# plt.colorbar()
# plt.title("Flow v")

# plt.figure()
# plt.imshow(np.linalg.norm(flow_up, axis=-1))
# plt.colorbar()
# plt.title("Flow Magnitude")

# print("mag shape:", mag.shape)
print("rgb 1 cropped shape:", rgb_1_cropped.shape)

# 4. Use the flow to, for each pixel, compute the index at time t + 1 from time t (just each pixel's
#    image space coordinate in image 1's transformed frame)
#     1. This would entail computing the dx, dy image space coordinate change from the flow's
#        magnitude and angle channels (output), then adding that to the meshgrid coordinates (the
#        coordinates at time t)
img_width = rgb_1_cropped.shape[1]
img_height = rgb_1_cropped.shape[0]
# v_coords_1, u_coords_1 = np.meshgrid(np.arange(img_width), np.arange(img_height))
u_coords_1, v_coords_1 = get_uv_coords(img_height, img_width)
u_coords_1 = u_coords_1.flatten()
v_coords_1 = v_coords_1.flatten()
z_vals_1 = depth_1_cropped.flatten()

# mag_flat = mag.flatten()

# TODO: Is this angle in the same coordinate frame as the image?
# e.g. might have to flip the angle, rotate by 90 degrees, etc.
# angle_flat = angle.flatten() / 180 * np.pi

# du = np.round(mag_flat * np.cos(angle_flat)).astype(int)
# dv = np.round(mag_flat * np.sin(angle_flat)).astype(int)
dv = np.round(flow_up[..., 0].flatten()).astype(int)
du = np.round(flow_up[..., 1].flatten()).astype(int)

# Subtracting coordinates since coordinate frames for flow and images are different
u_coords_2 = u_coords_1 + du
v_coords_2 = v_coords_1 + dv

# TODO: Cut out all coordinates that are out of bounds
# mask_valid_points = np.ones((img_height, img_width), dtype=bool)
coords_valid = np.ones_like(u_coords_2, dtype=bool)
# us_invalid = (u_coords_2 < 0) & (u_coords_2 >= img_width)
# vs_invalid = (v_coords_2 < 0) & (v_coords_2 >= img_height)
coords_valid[u_coords_2 < 0] = False
coords_valid[u_coords_2 >= img_width] = False
coords_valid[v_coords_2 < 0] = False
coords_valid[v_coords_2 >= img_height] = False
# coords_invalid = us_invalid | vs_invalid
# mask_valid_points[coords_invalid] = False

u_coords_2_valid = u_coords_2[coords_valid]
v_coords_2_valid = v_coords_2[coords_valid]

z_vals_2 = depth_2_cropped[v_coords_2_valid, u_coords_2_valid]
# z_vals_2 = depth_2_cropped[v_coords_2_valid, u_coords_2_valid]
# z_coords_2 = depth_2_cropped.flatten()

crop_shape = rgb_1_cropped.shape

print(crop_shape)

print("Image 1 shape:", rgb_1.shape)
print("u min crop:", valid_bbox.u_min)
print("u max crop:", valid_bbox.u_max)
print("v min crop:", valid_bbox.v_min)
print("v max crop:", valid_bbox.v_max)

u_coords_1_valid = u_coords_1[coords_valid]
v_coords_1_valid = v_coords_1[coords_valid]
z_vals_1_valid = z_vals_1[coords_valid]

# TODO: The intrinsic expects the UV coordinates to be generated based on images that are (480, 640)
# but due to how we're cropping things, the UV coordinates are generated based on the cropped images
# that could be quite different. We need to adjust the UV coordinates to be based on the original
# image size.
us_fixed = u_coords_1_valid + valid_bbox.u_min
vs_fixed = v_coords_1_valid + valid_bbox.v_min

xyz_1 = uvz_ocv_to_xyz_ned(u_coords_1_valid, v_coords_1_valid, z_vals_1_valid)
rgb_vals_1 = rgb_1_cropped[v_coords_1_valid, u_coords_1_valid]
xyz_2 = uvz_ocv_to_xyz_ned(u_coords_2_valid, v_coords_2_valid, z_vals_2)
rgb_vals_2 = rgb_2_cropped[v_coords_2_valid, u_coords_2_valid]

from limap_extension.point_cloud import PointCloud

cloud_1_in_frame_2 = PointCloud(xyz_1, rgb_vals_1)
cloud_2_in_frame_2 = PointCloud(xyz_2, rgb_vals_2)

rr.log("cloud_1_in_frame_2", rr.Points3D(**cloud_1_in_frame_2.form_rerun_kwargs(radii=0.02)))
rr.log("cloud_2_in_frame_2", rr.Points3D(**cloud_2_in_frame_2.form_rerun_kwargs(radii=0.02)))

# 5. Project these two images into a point cloud
# convert the point cloud using grid coordinates and depths
# Just use intrinsic
# xyz_1 = np.random.rand(IMG_HEIGHT * IMG_WIDTH, 3)
# xyz_2 = np.random.rand(IMG_HEIGHT * IMG_WIDTH, 3)

# 6. Index point cloud 1 with coordinate at time t, point cloud 2 with coordinate at time t +1,
# 7. Compute Euclidean distance
# flow_3d = np.linalg.norm(xyz_1 - xyz_2, axis=0)

# print("Flow min:", flow_3d.min())
# print("Flow max:", flow_3d.max())

# points_dynamic_mask = flow_3d > 0.05  # meters
# # coords_dynamic_1 = xyz_1[points_dynamic_mask, :]
# coords_dynamic_2 = xyz_2[:, points_dynamic_mask]

# # 8. reproject the points back into image space with the distance being the value at each pixel
# mask = np.zeros((img_height, img_width), dtype=np.uint8)
# # reproject to image coordinates using funciton and intrinsic
# us, vs, zs = xyz_to_uvz(coords_dynamic_2.T, is_rounding_to_int=True)

# from limap_extension.img_cloud_transforms import find_valid_uv_coords

# where_valid = find_valid_uv_coords(us, vs, img_height, img_width)
# us = us[where_valid]
# vs = vs[where_valid]

# mask[vs, us] = 1

# depth_reconstructed = np.zeros((img_height, img_width), dtype=np.float32)
# depth_reconstructed[vs, us] = zs

# threshold = 30
# depth_reconstructed[depth_reconstructed > threshold] = threshold
# plt.figure()
# plt.imshow(depth_reconstructed)

# plt.figure()
# plt.imshow(mask)
