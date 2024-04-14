from typing import Tuple
from warnings import warn
import copy

import numpy as np
import torch
# import cv2

from constants import CAM_INTRINSIC

# TartainAir depth unit is meters.
MILLIMETERS_TO_METERS = 1e-3
METERS_TO_METERS = 1.0


class PointCloud:

    def __init__(self, xyz: np.ndarray, rgb: np.ndarray):
        self.xyz: np.ndarray = self._verify_shape(xyz)
        if rgb.dtype != np.uint8:
            raise ValueError("RGB array must be of type uint8")
        self.rgb: np.ndarray = self._verify_shape(rgb)

    def _verify_shape(self, arr: np.ndarray) -> np.ndarray:
        if not arr.ndim == 2:
            raise ValueError("Array must be 2D")
        if not arr.shape[1] == 3:
            raise ValueError(f"Array convention is [N_points, 3], got {arr.shape} instead.")
        return arr


def get_uv_coords(img_rows, img_cols):
    V, U = np.meshgrid(np.arange(img_rows), np.arange(img_cols))
    V = V.T.flatten()
    U = U.T.flatten()
    return U, V


def imgs_to_clouds_np(rgb_img: np.ndarray,
                      depth_img: np.ndarray,
                      intrinsic: np.ndarray,
                      depth_units_to_tracked_units: float = METERS_TO_METERS) -> PointCloud:
    """Converts RGB/D images to point clouds represented by XYZ and RGB arrays

    This has a lot of commented code that can be used to filter out points given a 2D (segmentation)
    mask. I don't think we'll need this but I'm keeping it in case we do.
    """
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]

    center_x = intrinsic[0, 2]
    center_y = intrinsic[1, 2]

    constant_x = 1.0 / fx
    constant_y = 1.0 / fy

    U, V = get_uv_coords(depth_img.shape[0], depth_img.shape[1])

    depth_flat = depth_img.flatten()

    # Get individual values.
    xs = (U - center_x) * depth_flat * depth_units_to_tracked_units * constant_x
    ys = (V - center_y) * depth_flat * depth_units_to_tracked_units * constant_y
    zs = depth_flat * depth_units_to_tracked_units

    # Z = 0 indicates the point is invalid in the depth images that I've been working with in the
    # lab. I'm not sure if TartanAir has a similar convention.
    where_depth_valid = zs != 0.0

    # This can be used to apply a segmentation mask so that we get a segmented point cloud. I don't
    # think we'll need this but I'm keeping it in case we do.
    # where_keep_points = np.logical_and(mask_flat, where_depth_valid)
    # where_keep_points = where_depth_valid

    xyz_cloud_raw = np.stack((xs, ys, zs), axis=0)
    xyz_cloud_unfiltered = xyz_cloud_raw[:, where_depth_valid]
    # xyz_cloud_filtered = xyz_cloud_raw[:, where_keep_points]

    num_rgb_channels = rgb_img.shape[2]
    rgb_cloud_raw = rgb_img.reshape(-1, num_rgb_channels).T
    rgb_cloud_unfiltered = rgb_cloud_raw[:, where_depth_valid]
    # rgb_cloud_filtered = rgb_cloud_raw[:, where_keep_points]

    return PointCloud(xyz_cloud_unfiltered.T, rgb_cloud_unfiltered.T)


def cloud_to_img_np(cloud: PointCloud,
                    intrinsic: np.ndarray,
                    img_width: int = 640,
                    img_height: int = 480,
                    depth_units_to_tracked_units: float = METERS_TO_METERS) -> np.ndarray:
    """Turns cloud coordinates to UV (image) coordinates

    TODO: Do we need to do some sort of bilinear interpolation here to correct for camera
    distortions? While working with images from real sensors, I've seen that reprojection from 3D to
    2D rasterized images can leave holes in the image due to camera distortions.
    """
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]

    center_x = intrinsic[0, 2]
    center_y = intrinsic[1, 2]

    constant_x = 1.0 / fx
    constant_y = 1.0 / fy

    xs = cloud.xyz[:, 0]
    ys = cloud.xyz[:, 1]
    zs = cloud.xyz[:, 2] / depth_units_to_tracked_units

    vs = xs / (zs * depth_units_to_tracked_units * constant_x) + center_x
    us = ys / (zs * depth_units_to_tracked_units * constant_y) + center_y

    uv_coords = np.round(np.stack((us, vs), axis=0)).astype(int)

    where_u_valid = np.logical_and(uv_coords[0] >= 0, uv_coords[0] < img_height)
    where_v_valid = np.logical_and(uv_coords[1] >= 0, uv_coords[1] < img_width)
    where_uv_valid = np.logical_and(where_u_valid, where_v_valid)

    uv_coords = uv_coords[:, where_uv_valid]
    rgb_valid = cloud.rgb[where_uv_valid, :]

    warn("May want to use interpolation for cloud to image conversion. If you see holes in the "
         "image resulting from this conversion, implement interpolation scheme.")

    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    img[uv_coords[0], uv_coords[1], :] = rgb_valid

    # Old for loop implementation. Keeping this in case we need it for the intepolation we might
    # wind up doing to handle camera distortions.
    # for i in range(uv_coords.shape[1]):
    #     v, u = uv_coords[:, i]
    #     if v >= img_width or u >= img_height:
    #         print(f"({u}, {v}) out of bounds!")
    #         continue
    #     rgb = cloud.rgb[i, :]
    #     img[u, v, :] = rgb

    return img


def transform_cloud(cloud: PointCloud, H: np.ndarray) -> PointCloud:
    """Transforms a point coud from one frame to another using a homogeneous tform matrix"""
    cloud_tformed = copy.deepcopy(cloud)

    # Augment the xyz cloud with 1s so transform matmul works.
    xyz_aug = np.hstack([cloud_tformed.xyz, np.ones((cloud_tformed.xyz.shape[0], 1))])

    xyz_tformed = (H @ xyz_aug.T).T
    cloud_tformed.xyz = xyz_tformed[:, :-1]
    return cloud_tformed


def reproject_img(rgb_1: np.ndarray, depth_1: np.ndarray, pose_1: np.ndarray, pose_2: np.ndarray):
    """Reprojects an image from frame 1 to frame 2 using the poses of the two frames."""
    cloud_frame_1 = imgs_to_clouds_np(rgb_1, depth_1, CAM_INTRINSIC)
    H_1_2 = np.linalg.inv(pose_1) @ pose_2
    cloud_tformed = transform_cloud(cloud_frame_1, H_1_2)
    img_tformed = cloud_to_img_np(cloud_tformed, CAM_INTRINSIC)
    return img_tformed


def imgs_to_clouds_torch(
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor,
        intrinsic: torch.Tensor,
        mask: torch.Tensor = None,
        depth_units_to_tracked_units: float = METERS_TO_METERS) -> Tuple[PointCloud, PointCloud]:
    """Pretty faithful copy of the C++ image to cloud conversion routine"""
    raise NotImplementedError("This function is not yet implemented for PyTorch tensors.")
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]

    center_x = intrinsic[0, 2]
    center_y = intrinsic[1, 2]

    constant_x = 1.0 / (fx * PIXEL_LEN)
    constant_y = 1.0 / (fy * PIXEL_LEN)

    U, V = get_uv_coords(rgb_img.shape[0], rgb_img.shape[1])
    U = torch.from_numpy(U)
    V = torch.from_numpy(V)

    depth_flat = depth_img.flatten()

    # Get individual values.
    xs = (U - center_x) * PIXEL_LEN * depth_flat * depth_units_to_tracked_units * constant_x
    ys = (V - center_y) * PIXEL_LEN * depth_flat * depth_units_to_tracked_units * constant_y
    zs = depth_flat * depth_units_to_tracked_units

    if mask is not None:
        mask_flat = mask.flatten()
    else:
        mask_flat = torch.ones_like(depth_flat, dtype=bool, device=rgb_img.device)

    # Z = 0 indicates the point is invalid.
    where_depth_valid = zs != 0.0
    where_keep_points = torch.logical_and(mask_flat, where_depth_valid)

    rgb_cloud_raw = rgb_img.reshape(-1, 3).T
    xyz_cloud_raw = torch.stack((xs, ys, zs), dim=0)

    rgb_cloud_unfiltered = rgb_cloud_raw[:, where_depth_valid]
    xyz_cloud_unfiltered = xyz_cloud_raw[:, where_depth_valid]

    rgb_cloud_filtered = rgb_cloud_raw[:, where_keep_points]
    xyz_cloud_filtered = xyz_cloud_raw[:, where_keep_points]

    cloud_unfiltered = PointCloud(xyz_cloud_unfiltered, rgb_cloud_unfiltered)
    cloud_filtered = PointCloud(xyz_cloud_filtered, rgb_cloud_filtered)

    return cloud_unfiltered, cloud_filtered


# NOTE(dylan.colli): The rest of these functions are from the CDCPD file I ripped the image/point
# cloud utilities from. I don't think we need this but I'll keep it just in case.

# def segment_hsv(bgr: np.ndarray):
#     """Naive HSV segmentation like what is done in the CDCPD C++ implementation"""
#     if np.max(bgr) > 1.0:
#         bgr = bgr / 255.
#     if bgr.dtype != np.float32:
#         bgr = bgr.astype(np.float32)
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
#     hue_min = 340.0
#     sat_min = 0.3
#     val_min = 0.4
#     sat_max = 1.0
#     val_max = 1.0
#     hue_max2 = 360.0
#     hue_min1 = 0.0
#     hue_max = 20
#     mask1 = cv2.inRange(hsv, (hue_min, sat_min, val_min), (hue_max2, sat_max, val_max))
#     mask2 = cv2.inRange(hsv, (hue_min1, sat_min, val_min), (hue_max, sat_max, val_max))
#     mask = np.logical_or(mask1, mask2)
#     return mask

# def get_bounding_box(Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     # last lower bounding box and last upper bounding box
#     llbb = Y.min(dim=1, keepdim=True).values
#     lubb = Y.max(dim=1, keepdim=True).values
#     # print(llbb, lubb)
#     return llbb, lubb

# def get_bounding_box_np(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     # last lower bounding box and last upper bounding box
#     llbb = Y.min(axis=1).reshape(3, 1)
#     lubb = Y.max(axis=1).reshape(3, 1)
#     # print(llbb, lubb)
#     return llbb, lubb

# def apply_bounding_box(llbb: torch.Tensor,
#                        lubb: torch.Tensor,
#                        cloud: PointCloud,
#                        bbox_extend: float = 0.1,
#                        verbose: bool = False):
#     mask = torch.ones(cloud.xyz.shape[1], dtype=bool, device=llbb.device)
#     for i in range(3):
#         ax_mask = torch.logical_and(cloud.xyz[i, :] > (llbb[i] - bbox_extend), cloud.xyz[i, :]
#                                     < (lubb[i] + bbox_extend))
#         mask = torch.logical_and(ax_mask, mask)

#     num_points_pre = cloud.xyz.shape[1]
#     cloud.xyz = cloud.xyz[:, mask]
#     num_points_post = cloud.xyz.shape[1]
#     if verbose:
#         print(f"Filtered from {num_points_pre} -> {num_points_post}")

#     if cloud.has_rgb:
#         cloud.rgb = cloud.rgb[:, mask]

# def apply_bounding_box_np(llbb: np.ndarray,
#                           lubb: np.ndarray,
#                           cloud: PointCloud,
#                           bbox_extend: float = 0.1,
#                           verbose: bool = False):
#     mask = np.ones(cloud.xyz.shape[1], dtype=bool)
#     for i in range(3):
#         ax_mask = np.logical_and(cloud.xyz[i, :] > (llbb[i] - bbox_extend), cloud.xyz[i, :]
#                                  < (lubb[i] + bbox_extend))
#         mask = np.logical_and(ax_mask, mask)

#     num_points_pre = cloud.xyz.shape[1]
#     cloud.xyz = cloud.xyz[:, mask]
#     num_points_post = cloud.xyz.shape[1]
#     if verbose:
#         print(f"Filtered from {num_points_pre} -> {num_points_post}")

#     if cloud.has_rgb:
#         cloud.rgb = cloud.rgb[:, mask]

# def bbox_filter(Y: torch.Tensor, cloud: PointCloud, bbox_extend: float = 0.1):
#     llbb, lubb = get_bounding_box(Y)
#     apply_bounding_box(llbb, lubb, cloud, bbox_extend=bbox_extend)

# def bbox_filter_np(Y: np.ndarray, cloud: np.ndarray, bbox_extend: float = 0.1):
#     llbb, lubb = get_bounding_box_np(Y)
#     apply_bounding_box_np(llbb, lubb, cloud, bbox_extend=bbox_extend)
