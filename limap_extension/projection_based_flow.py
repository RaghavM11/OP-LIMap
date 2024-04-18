import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2

REPO_DIR = Path(".").resolve().parents[0]
# print(REPO_DIR)
sys.path.append(REPO_DIR.as_posix())

from limap_extension.optical_flow import Args, OpticalFlow, RAFT_MODEL_PATH
from limap_extension.img_cloud_transforms import (reproject_img, uvz_ned_to_xyz_cam,
                                                  xyz_cam_to_uvz_ned, index_img_with_uv_coords,
                                                  get_uv_coords, imgs_to_clouds_np,
                                                  find_valid_uv_coords)
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array
from limap_extension.bounding_box import BoundingBox

# from limap_extension.visualization.rerun.figure_factory import FigureFactory


def display_img_pair(rgb, depth, img_slice: BoundingBox = None):
    if img_slice is not None:
        rgb = img_slice.crop_img(rgb)
        depth = img_slice.crop_img(depth)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)

    # depth = np.clip(depth, 0, 10)
    im1 = ax[1].imshow(depth)
    fig.colorbar(im1, ax=ax[1])


def project_img_pair_to_3d_using_flow(rgb_1_cropped: np.ndarray, depth_1_cropped: np.ndarray,
                                      depth_2_cropped: np.ndarray, flow_up,
                                      mask_valid_projection_cropped):
    # Use the flow to, for each pixel, compute the index at time t + 1 from time t (just each pixel's
    # image space coordinate in image 1's transformed frame)
    # - This would entail computing the dx, dy image space coordinate change from the flow's magnitude
    #   and angle channels (output), then adding that to the meshgrid coordinates (the coordinates at
    #   time t)
    img_width = rgb_1_cropped.shape[1]
    img_height = rgb_1_cropped.shape[0]
    # v_coords_1, u_coords_1 = np.meshgrid(np.arange(img_width), np.arange(img_height))
    u_coords_1, v_coords_1 = get_uv_coords(img_height, img_width)
    u_coords_1 = u_coords_1.flatten()
    v_coords_1 = v_coords_1.flatten()
    z_vals_1 = depth_1_cropped.flatten()

    dv = np.round(flow_up[..., 0].flatten()).astype(int)
    du = np.round(flow_up[..., 1].flatten()).astype(int)

    # Subtracting coordinates since coordinate frames for flow and images are different
    u_coords_2 = u_coords_1 + du
    v_coords_2 = v_coords_1 + dv

    coords_valid = find_valid_uv_coords(u_coords_2, v_coords_2, img_height, img_width)

    proj_valid = mask_valid_projection_cropped.flatten()
    coords_valid = coords_valid & proj_valid

    u_coords_2_valid = u_coords_2[coords_valid]
    v_coords_2_valid = v_coords_2[coords_valid]

    z_vals_2 = depth_2_cropped[v_coords_2_valid, u_coords_2_valid]

    # crop_shape = rgb_1_cropped.shape
    # print("Image 1 shape:", rgb_1.shape)
    # print("u min crop:", valid_bbox.u_min)
    # print("u max crop:", valid_bbox.u_max)
    # print("v min crop:", valid_bbox.v_min)
    # print("v max crop:", valid_bbox.v_max)

    u_coords_1_valid = u_coords_1[coords_valid]
    v_coords_1_valid = v_coords_1[coords_valid]
    z_vals_1_valid = z_vals_1[coords_valid]

    # The intrinsic expects the UV coordinates to be generated based on images that are (480, 640)
    # but due to how we're cropping things, the UV coordinates are generated based on the cropped images
    # that could be quite different. We need to adjust the UV coordinates to be based on the original
    # image size.
    # - Actually, since both images are cropped the same way, we don't HAVE to do this, but it means
    #   that the projections will be off.
    # NOTE: If registration looks off, this is the first place to check.
    # us_fixed = u_coords_1_valid + valid_bbox.u_min
    # vs_fixed = v_coords_1_valid + valid_bbox.v_min

    xyz_1 = uvz_ned_to_xyz_cam(u_coords_1_valid, v_coords_1_valid, z_vals_1_valid)
    xyz_2 = uvz_ned_to_xyz_cam(u_coords_2_valid, v_coords_2_valid, z_vals_2)

    return xyz_1, xyz_2


def calculate_flow_field(xyz_1: np.ndarray, xyz_2: np.ndarray) -> np.ndarray:

    # z_disparity = z_vals_1_valid - z_vals_2

    # print("Min disparity:", z_disparity.min())
    # print("Max disparity:", z_disparity.max())

    # z_disparity[z_disparity < -2] = -2
    # z_disparity[z_disparity > 2] = 2

    # plt.figure()
    # plt.plot(z_disparity[:3000])

    # 6. Index point cloud 1 with coordinate at time t, point cloud 2 with coordinate at time t +1,
    # 7. Compute Euclidean distance
    flow_3d = np.linalg.norm(xyz_1 - xyz_2, axis=1)

    # print("Flow min:", flow_3d.min())
    # print("Flow max:", flow_3d.max())

    return flow_3d


def segment_projection_based_flow(flow_3d: np.ndarray, xyz_2: np.ndarray, img_height: int,
                                  img_width: int) -> np.ndarray:
    # TODO: Consider cleaning and clustering the naive threshold-based mask then indexing the ground
    # truth segmentation mask to make a very clean dynamic object mask.

    points_dynamic_mask = flow_3d > 0.5  # meters
    # coords_dynamic_1 = xyz_1[points_dynamic_mask, :]
    coords_dynamic_2 = xyz_2[points_dynamic_mask, :]

    # Reproject the points back into image space with the distance being the value at each pixel
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    us, vs, zs = xyz_cam_to_uvz_ned(coords_dynamic_2, is_rounding_to_int=True)

    where_valid = find_valid_uv_coords(us, vs, img_height, img_width)
    us = us[where_valid]
    vs = vs[where_valid]

    mask[vs, us] = 1

    # display_img_pair(rgb_1_cropped, rgb_2_cropped)

    # plt.figure()
    # plt.imshow(mask)

    depth_reconstructed = np.zeros((img_height, img_width), dtype=np.float32)
    depth_reconstructed[vs, us] = zs

    # threshold = 30
    # depth_reconstructed[depth_reconstructed > threshold] = threshold
    # plt.figure()
    # plt.imshow(depth_reconstructed)

    print("Do we need reconstructed depth? Could give us an idea of how well the projection is "
          "working.")

    return mask, depth_reconstructed


def projection_based_motion_segmentation(rgb_1: np.ndarray, depth_1: np.ndarray, rgb_2: np.ndarray,
                                         depth_2: np.ndarray, pose_1: np.ndarray,
                                         pose_2: np.ndarray, flow: OpticalFlow):
    # Reproject the image at time t to the image frame at time t+1
    img_1_in_frame_2, depth_1_in_frame_2, mask_valid_projection, valid_bbox = reproject_img(
        rgb_1, depth_1, pose_1, pose_2)

    # display_img_pair(img_1_in_frame_2, depth_1_in_frame_2)

    # valid_bbox = BoundingBox(0, -1, 0, -1)

    # print(valid_bbox)

    rgb_1_cropped = valid_bbox.crop_img(img_1_in_frame_2)
    depth_1_cropped = valid_bbox.crop_img(depth_1_in_frame_2)
    rgb_2_cropped = valid_bbox.crop_img(rgb_2)
    depth_2_cropped = valid_bbox.crop_img(depth_2)
    mask_valid_projection_cropped = valid_bbox.crop_img(mask_valid_projection)

    img_height_cropped = rgb_1_cropped.shape[0]
    img_width_cropped = rgb_1_cropped.shape[1]

    # TODO: Make this a callable input to the function that either actually calculates the flow or
    # loads in the ground truth flow?
    # Might be easier to make this a flag and conditional.
    _, flow_up = flow.infer_flow(rgb_1_cropped, rgb_2_cropped)
    flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()

    xyz_1, xyz_2 = project_img_pair_to_3d_using_flow(rgb_1_cropped, depth_1_cropped,
                                                     depth_2_cropped, flow_up,
                                                     mask_valid_projection_cropped)

    flow_field = calculate_flow_field(xyz_1, xyz_2)

    mask = segment_projection_based_flow(flow_field, xyz_2, img_height_cropped, img_width_cropped)

    return mask


if __name__ == "__main__":
    # TRIAL_DIR = REPO_DIR / "datasets" / "ocean" / "Hard" / "P006"
    TRIAL_DIR = REPO_DIR / "datasets" / "carwelding" / "easy" / "P007"
    # FRAME_1 = 550
    # FRAME_2 = 551
    FRAME_1 = 127
    FRAME_2 = 128

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

    flow = OpticalFlow(None)
    flow.load_model(RAFT_MODEL_PATH, Args())

    cp1 = cam_poses[FRAME_1, :]
    cp2 = cam_poses[FRAME_2, :]

    pose_1 = get_transform_matrix_from_pose_array(cp1)
    pose_2 = get_transform_matrix_from_pose_array(cp2)

    # cloud_1, _ = imgs_to_clouds_np(rgb_1, depth_1, CAM_INTRINSIC)
