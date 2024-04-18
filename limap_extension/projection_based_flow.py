import sys
from pathlib import Path
from typing import Tuple

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
from limap_extension.point_cloud import PointCloud

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


def preprocess_valid_projection_mask(mask_valid_proj: np.ndarray):
    # Do some processing to invalidate the areas around the projection mask. This deals with
    # repeated, thin, sharp objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_adjusted = cv2.morphologyEx(mask_valid_proj.astype(np.uint8),
                                     cv2.MORPH_OPEN,
                                     kernel,
                                     iterations=5).astype(bool)
    return mask_adjusted


# def project_img_pair_to_3d_using_flow(
#         img_1_in_frame_2_cropped: np.ndarray, depth_1_in_frame_2_cropped: np.ndarray,
#         depth_2_cropped: np.ndarray, flow_up,
#         mask_valid_projection_cropped) -> Tuple[PointCloud, PointCloud]:
#     # # Use the flow to, for each pixel, compute the index at time t + 1 from time t (just each pixel's
#     # # image space coordinate in image 1's transformed frame)
#     # # - This would entail computing the dx, dy image space coordinate change from the flow's magnitude
#     # #   and angle channels (output), then adding that to the meshgrid coordinates (the coordinates at
#     # #   time t)
#     # # NOTE: unrelated to above. But I think the super-ultra-rigorous way to do this would be to
#     # # estimate the depth flow based on the flow contraction and expansion at each point, then to use
#     # # this depth flow to estimate the 3D flow given flow in U and V coordinates as well. Maybe we
#     # # can mention this in the report as a limitation.
#     # # - Actually, this is an area of research called "scene flow". We probably should have used that
#     # #   but oh well.
#     # # TODO: Should we actually be producing 3 point clouds? Seems like we need img_1_in_frame_2
#     # # cloud, predicted t = t_2 cloud given flow-based projection, and the actual t = t_2 cloud.
#     # img_width = img_1_in_frame_2_cropped.shape[1]
#     # img_height = img_1_in_frame_2_cropped.shape[0]

#     # us_1_in_frame_2, vs_1_in_frame_2 = get_uv_coords(img_height, img_width)
#     # us_1_in_frame_2 = us_1_in_frame_2.flatten()
#     # vs_1_in_frame_2 = vs_1_in_frame_2.flatten()

#     # z_vals_1_in_frame_2 = depth_1_in_frame_2_cropped.flatten()
#     # rgb_vals_1_in_frame_2 = img_1_in_frame_2_cropped.reshape(-1, 3)

#     # dv = np.round(flow_up[..., 0].flatten()).astype(int)
#     # du = np.round(flow_up[..., 1].flatten()).astype(int)

#     # # Subtracting coordinates since coordinate frames for flow and images are different
#     # u_coords_2_predicted = us_1_in_frame_2 + du
#     # v_coords_2_predicted = vs_1_in_frame_2 + dv

#     # coords_valid = find_valid_uv_coords(u_coords_2_predicted, v_coords_2_predicted, img_height,
#     #                                     img_width)

#     # proj_valid = mask_valid_projection_cropped.flatten()
#     # coords_valid = coords_valid & proj_valid

#     # u_coords_2_valid = u_coords_2_predicted[coords_valid]
#     # v_coords_2_valid = v_coords_2_predicted[coords_valid]

#     # z_vals_2_predicted = index_img_with_uv_coords(u_coords_2_valid, v_coords_2_valid,
#     #                                               depth_2_cropped)

#     # z_vals_2 = index_img_with_uv_coords(u_coords_2_valid, v_coords_2_valid, depth_2_cropped)

#     # u_coords_1_valid = us_1_in_frame_2[coords_valid]
#     # v_coords_1_valid = vs_1_in_frame_2[coords_valid]
#     # z_vals_1_valid = z_vals_1_in_frame_2[coords_valid]
#     # rgb_vals_1_valid = rgb_vals_1_in_frame_2[coords_valid]

#     # # Since we ultimately want to reconstruct the images, we use the RGB values from the first image
#     # # using the transformed coordinates (based on flow). This is a roundabout way to warp the image
#     # # given the calculated flow.
#     # rgb_vals_2_valid = rgb_vals_1_valid

#     # # The intrinsic expects the UV coordinates to be generated based on images that are (480, 640)
#     # # but due to how we're cropping things, the UV coordinates are generated based on the cropped images
#     # # that could be quite different. We need to adjust the UV coordinates to be based on the original
#     # # image size.
#     # # - Actually, since both images are cropped the same way, we don't HAVE to do this, but it means
#     # #   that the projections will be off.
#     # # NOTE: If registration looks off, this is the first place to check.
#     # # us_fixed = u_coords_1_valid + valid_bbox.u_min
#     # # vs_fixed = v_coords_1_valid + valid_bbox.v_min

#     # xyz_2_predicted = uvz_ned_to_xyz_cam(u_coords_1_valid, v_coords_1_valid, z_vals_1_valid)
#     # cloud_2_predicted = PointCloud(xyz_2_predicted, rgb_vals_1_valid)

#     # xyz_2_actual = uvz_ned_to_xyz_cam(u_coords_2_valid, v_coords_2_valid, z_vals_2)
#     # cloud_2_actual = PointCloud(xyz_2_actual, rgb_vals_2_valid)

#     # return cloud_2_predicted, cloud_2_actual

#     # proj_mask_to_use = mask_valid_projection_cropped
#     proj_mask_to_use = preprocess_valid_projection_mask(mask_valid_projection_cropped)

#     us, vs = get_uv_coords(*flow_up.shape[:-1])
#     dus = flow_up[:, :, 1].flatten()
#     dvs = flow_up[:, :, 0].flatten()
#     zs_1_in_frame_2 = depth_1_in_frame_2_cropped.flatten()
#     zs_2 = depth_2_cropped.flatten()

#     img_height, img_width, _ = flow_up.shape

#     u_preds = np.round(us + dus).astype(int)
#     v_preds = np.round(vs + dvs).astype(int)
#     coords_valid = find_valid_uv_coords(u_preds, v_preds, img_height, img_width)
#     coords_valid = coords_valid & proj_mask_to_use.flatten()

#     us = us[coords_valid]
#     vs = vs[coords_valid]
#     u_preds = u_preds[coords_valid]
#     v_preds = v_preds[coords_valid]
#     zs_1_in_frame_2 = zs_1_in_frame_2[coords_valid]
#     zs_2 = zs_2[coords_valid]

#     xyz_1_in_frame_2 = uvz_ned_to_xyz_cam(us, vs, zs_1_in_frame_2)
#     # NOTE: This isn't actually truly the predicted XYZ since we don't have depth-based flow.
#     # However, it seems good enough.
#     xyz_flow_based_pred = uvz_ned_to_xyz_cam(u_preds, v_preds, zs_2)

#     return PointCloud(xyz_1_in_frame_2), PointCloud(xyz_flow_based_pred)

# def calculate_flow_field_and_reconstruct_imgs(cloud_1: PointCloud, cloud_2: PointCloud,
#                                               img_height: int, img_width: int) -> np.ndarray:
#     """Calculates the 3D flow and projects this back to image space in frame 2"""
#     flow_3d = np.linalg.norm(cloud_1.xyz - cloud_2.xyz, axis=1)

#     # Reproject the points back into image space with the measured 3D flow being the value at each
#     # pixel in the flow field
#     flow_field = np.zeros((img_height, img_width), dtype=float)

#     # Reconstructed images could be useful for measuring where the flow prediction was accurate so
#     # I'm going to compute and return it.
#     depth_reconstructed = np.zeros((img_height, img_width), dtype=float)
#     rgb_reconstructed = np.zeros((img_height, img_width, 3), dtype=np.uint8)
#     mask_valid_reconstruction = np.zeros((img_height, img_width), dtype=bool)

#     us, vs, zs = xyz_cam_to_uvz_ned(cloud_2.xyz, is_rounding_to_int=True)

#     where_valid = find_valid_uv_coords(us, vs, img_height, img_width)
#     us = us[where_valid]
#     vs = vs[where_valid]

#     flow_field[vs, us] = flow_3d[where_valid]
#     depth_reconstructed[vs, us] = zs[where_valid]
#     rgb_reconstructed[vs, us] = cloud_2.rgb[where_valid]
#     mask_valid_reconstruction[vs, us] = True

#     return flow_field, rgb_reconstructed, depth_reconstructed, mask_valid_reconstruction


def flow_xyz_from_decomposed_motion(flow_up: np.ndarray, depth_1_cropped: np.ndarray,
                                    depth_2_cropped: np.ndarray, mask_valid_projection_cropped):
    # Idea: decompose the flow field into planar and depth components.
    proj_mask_to_use = preprocess_valid_projection_mask(mask_valid_projection_cropped)
    us, vs = get_uv_coords(*flow_up.shape[:-1])
    dus = flow_up[:, :, 1].flatten()
    dvs = flow_up[:, :, 0].flatten()
    zs_1_in_frame_2 = depth_1_cropped.reshape(-1)
    # zs_2 = depth_2_cropped.reshape(-1)

    xyz_1 = uvz_ned_to_xyz_cam(us, vs, zs_1_in_frame_2)
    xyz_2 = uvz_ned_to_xyz_cam(us + dus, vs + dvs, zs_1_in_frame_2)

    delta_xy = (xyz_1 - xyz_2)[:, :-1]
    planar_motion = delta_xy.reshape(*flow_up.shape)

    print("Should we mask out the invalid projection motion?")

    x_ned_motion = planar_motion[:, :, 0]
    y_ned_motion = planar_motion[:, :, 1]

    # Now, depth distance can be calculated by the difference in depth between the two frames.
    depth_motion = np.abs(depth_2_cropped - depth_1_cropped)

    flow_xyz = np.stack((x_ned_motion, y_ned_motion, depth_motion), axis=-1)
    flow_xyz[~proj_mask_to_use] = 0.0

    return flow_xyz


def segment_flow_xyz(flow_xyz: np.ndarray, threshold: float = 0.4):
    flow_mag = np.linalg.norm(flow_xyz, axis=-1)
    flow_mask = flow_mag > threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    flow_mask_adjusted = flow_mask.astype(np.uint8)
    flow_mask_adjusted = cv2.erode(flow_mask_adjusted, kernel)
    flow_mask_adjusted = cv2.morphologyEx(flow_mask_adjusted, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    flow_mask_adjusted = cv2.dilate(flow_mask_adjusted, kernel)

    return flow_mask_adjusted.astype(bool)


# def segment_flow_field_basic(flow_field: np.ndarray, motion_threshold: float = 0.5) -> np.ndarray:
#     # TODO: Consider cleaning and clustering the naive threshold-based mask then indexing the ground
#     # truth segmentation mask to make a very clean dynamic object mask.

#     # This is kind of huge and indicates that we're having difficulty with thin objects/noise. How
#     # to attenuate?
#     points_dynamic_mask = flow_field > motion_threshold  # meters

#     return points_dynamic_mask

# def projection_based_motion_segmentation(rgb_1: np.ndarray, depth_1: np.ndarray, rgb_2: np.ndarray,
#                                          depth_2: np.ndarray, pose_1: np.ndarray,
#                                          pose_2: np.ndarray, flow: OpticalFlow):
#     # Reproject the image at time t to the image frame at time t+1
#     img_1_in_frame_2, depth_1_in_frame_2, mask_valid_projection, valid_bbox = reproject_img(
#         rgb_1, depth_1, pose_1, pose_2)

#     img_1_in_frame_2_cropped = valid_bbox.crop_img(img_1_in_frame_2)
#     depth_1_in_frame_2_cropped = valid_bbox.crop_img(depth_1_in_frame_2)
#     rgb_2_cropped = valid_bbox.crop_img(rgb_2)
#     depth_2_cropped = valid_bbox.crop_img(depth_2)
#     mask_valid_projection_cropped = valid_bbox.crop_img(mask_valid_projection)

#     img_height_cropped = img_1_in_frame_2_cropped.shape[0]
#     img_width_cropped = img_1_in_frame_2_cropped.shape[1]

#     # TODO: Make this a callable input to the function that either actually calculates the flow or
#     # loads in the ground truth flow?
#     # Might be easier to make this a flag and conditional.
#     _, flow_up = flow.infer_flow(img_1_in_frame_2_cropped, rgb_2_cropped)
#     flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()

#     cloud_1, cloud_2 = project_img_pair_to_3d_using_flow(img_1_in_frame_2_cropped,
#                                                          depth_1_in_frame_2_cropped,
#                                                          depth_2_cropped, flow_up,
#                                                          mask_valid_projection_cropped)

#     flow_field, rgb_reconstructed, depth_reconstructed, mask_valid_reconstruction = \
#         calculate_flow_field_and_reconstruct_imgs(
#         cloud_1, cloud_2, img_height_cropped, img_width_cropped)

#     mask = segment_flow_field_basic(flow_field)

#     return mask


def projection_based_motion_segmentation(rgb_1: np.ndarray, depth_1: np.ndarray, rgb_2: np.ndarray,
                                         depth_2: np.ndarray, pose_1: np.ndarray,
                                         pose_2: np.ndarray, flow: OpticalFlow):
    img_dims_orig = depth_1.shape
    # Reproject the image at time t to the image frame at time t+1
    img_1_in_frame_2, depth_1_in_frame_2, mask_valid_projection, valid_bbox = reproject_img(
        rgb_1, depth_1, pose_1, pose_2)

    img_1_in_frame_2_cropped = valid_bbox.crop_img(img_1_in_frame_2)
    depth_1_in_frame_2_cropped = valid_bbox.crop_img(depth_1_in_frame_2)
    rgb_2_cropped = valid_bbox.crop_img(rgb_2)
    depth_2_cropped = valid_bbox.crop_img(depth_2)
    mask_valid_projection_cropped = valid_bbox.crop_img(mask_valid_projection)

    img_height_cropped = img_1_in_frame_2_cropped.shape[0]
    img_width_cropped = img_1_in_frame_2_cropped.shape[1]

    # TODO: Make this a callable input to the function that either actually calculates the flow or
    # loads in the ground truth flow?
    # Might be easier to make this a flag and conditional.
    _, flow_up = flow.infer_flow(img_1_in_frame_2_cropped, rgb_2_cropped)
    flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()

    # cloud_1, cloud_2 = project_img_pair_to_3d_using_flow(img_1_in_frame_2_cropped,
    #                                                      depth_1_in_frame_2_cropped,
    #                                                      depth_2_cropped, flow_up,
    #                                                      mask_valid_projection_cropped)
    flow_xyz = flow_xyz_from_decomposed_motion(flow_up, depth_1_in_frame_2_cropped, depth_2_cropped,
                                               mask_valid_projection_cropped)

    # flow_field, rgb_reconstructed, depth_reconstructed, mask_valid_reconstruction = \
    #     calculate_flow_field_and_reconstruct_imgs(
    #     cloud_1, cloud_2, img_height_cropped, img_width_cropped)

    # mask = segment_flow_field_basic(flow_field)
    mask = segment_flow_xyz(flow_xyz)

    mask = valid_bbox.uncrop_img(mask, *img_dims_orig, fill_value=False)

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

    mask = projection_based_motion_segmentation(rgb_1, depth_1, rgb_2, depth_2, pose_1, pose_2,
                                                flow)
