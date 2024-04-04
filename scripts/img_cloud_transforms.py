from typing import Tuple

import numpy as np
import torch
import cv2

from arm_clouds import PointCloud

MILLIMETERS_TO_METERS = 1e-3
# I don't know why this is included in the original C++ routine and it ultimately cancels out in the
# point cloud calculation so I'm just leaving it.
PIXEL_LEN = 0.0002645833


def get_uv_coords(img_rows, img_cols):
    V, U = np.meshgrid(np.arange(img_rows), np.arange(img_cols))
    V = V.T.flatten()
    U = U.T.flatten()
    return U, V


def imgs_to_clouds_np(
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        intrinsic: np.ndarray,
        mask: np.ndarray = None,
        depth_units_to_tracked_units: float = MILLIMETERS_TO_METERS
) -> Tuple[PointCloud, PointCloud]:
    """Pretty faithful copy of the C++ image to cloud conversion routine

    NOTE: The PIXEL_LEN scalar actually gets cancelled out in the calculations but I wanted to keep
    everything as close to the original C++ routine as possible so I didn't take it out.

    Returns
    -------
    (PointCloud, PointCloud)
        Tuple where the first entry is the unfiltered cloud and the second entry is the filtered
        cloud if a mask was given to this function.
    """
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]

    center_x = intrinsic[0, 2]
    center_y = intrinsic[1, 2]

    constant_x = 1.0 / (fx * PIXEL_LEN)
    constant_y = 1.0 / (fy * PIXEL_LEN)

    U, V = get_uv_coords(depth_img.shape[0], depth_img.shape[1])

    depth_flat = depth_img.flatten()

    # Get individual values.
    xs = (U - center_x) * PIXEL_LEN * depth_flat * depth_units_to_tracked_units * constant_x
    ys = (V - center_y) * PIXEL_LEN * depth_flat * depth_units_to_tracked_units * constant_y
    zs = depth_flat * depth_units_to_tracked_units

    if mask is not None:
        mask_flat = mask.flatten()
    else:
        mask_flat = np.ones_like(depth_flat, dtype=bool)

    # Z = 0 indicates the point is invalid.
    where_depth_valid = zs != 0.0
    where_keep_points = np.logical_and(mask_flat, where_depth_valid)

    xyz_cloud_raw = np.stack((xs, ys, zs), axis=0)
    xyz_cloud_unfiltered = xyz_cloud_raw[:, where_depth_valid]
    xyz_cloud_filtered = xyz_cloud_raw[:, where_keep_points]

    unfiltered_cloud_kwargs = {"xyz": xyz_cloud_unfiltered}
    filtered_cloud_kwargs = {"xyz": xyz_cloud_filtered}

    if rgb_img is not None:
        num_rgb_channels = rgb_img.shape[2]
        rgb_cloud_raw = rgb_img.reshape(-1, num_rgb_channels).T
        unfiltered_cloud_kwargs["rgb"] = rgb_cloud_raw[:, where_depth_valid]
        filtered_cloud_kwargs["rgb"] = rgb_cloud_raw[:, where_keep_points]

    cloud_unfiltered = PointCloud(**unfiltered_cloud_kwargs)
    cloud_filtered = PointCloud(**filtered_cloud_kwargs)

    return cloud_unfiltered, cloud_filtered


def cloud_to_img_np(cloud: PointCloud,
                    intrinsic: np.ndarray,
                    depth_units_to_tracked_units: float = MILLIMETERS_TO_METERS):
    """Turns cloud coordinates to UV coordinates"""
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]

    center_x = intrinsic[0, 2]
    center_y = intrinsic[1, 2]

    constant_x = 1.0 / (fx * PIXEL_LEN)
    constant_y = 1.0 / (fy * PIXEL_LEN)

    xs = cloud.xyz[0, :]
    ys = cloud.xyz[1, :]
    zs = cloud.xyz[2, :] / depth_units_to_tracked_units

    us = xs / (PIXEL_LEN * zs * depth_units_to_tracked_units * constant_x) + center_x
    vs = ys / (PIXEL_LEN * zs * depth_units_to_tracked_units * constant_y) + center_y

    uv_coords = np.round(np.stack((us, vs), axis=0)).astype(int)

    return uv_coords, zs


def imgs_to_clouds_torch(
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor,
        intrinsic: torch.Tensor,
        mask: torch.Tensor = None,
        depth_units_to_tracked_units: float = MILLIMETERS_TO_METERS
) -> Tuple[PointCloud, PointCloud]:
    """Pretty faithful copy of the C++ image to cloud conversion routine"""
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


def segment_hsv(bgr: np.ndarray):
    """Naive HSV segmentation like what is done in the CDCPD C++ implementation"""
    if np.max(bgr) > 1.0:
        bgr = bgr / 255.
    if bgr.dtype != np.float32:
        bgr = bgr.astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
    hue_min = 340.0
    sat_min = 0.3
    val_min = 0.4
    sat_max = 1.0
    val_max = 1.0
    hue_max2 = 360.0
    hue_min1 = 0.0
    hue_max = 20
    mask1 = cv2.inRange(hsv, (hue_min, sat_min, val_min), (hue_max2, sat_max, val_max))
    mask2 = cv2.inRange(hsv, (hue_min1, sat_min, val_min), (hue_max, sat_max, val_max))
    mask = np.logical_or(mask1, mask2)
    return mask


def get_bounding_box(Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # last lower bounding box and last upper bounding box
    llbb = Y.min(dim=1, keepdim=True).values
    lubb = Y.max(dim=1, keepdim=True).values
    # print(llbb, lubb)
    return llbb, lubb


def get_bounding_box_np(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # last lower bounding box and last upper bounding box
    llbb = Y.min(axis=1).reshape(3, 1)
    lubb = Y.max(axis=1).reshape(3, 1)
    # print(llbb, lubb)
    return llbb, lubb


def apply_bounding_box(llbb: torch.Tensor,
                       lubb: torch.Tensor,
                       cloud: PointCloud,
                       bbox_extend: float = 0.1,
                       verbose: bool = False):
    mask = torch.ones(cloud.xyz.shape[1], dtype=bool, device=llbb.device)
    for i in range(3):
        ax_mask = torch.logical_and(cloud.xyz[i, :] > (llbb[i] - bbox_extend), cloud.xyz[i, :]
                                    < (lubb[i] + bbox_extend))
        mask = torch.logical_and(ax_mask, mask)

    num_points_pre = cloud.xyz.shape[1]
    cloud.xyz = cloud.xyz[:, mask]
    num_points_post = cloud.xyz.shape[1]
    if verbose:
        print(f"Filtered from {num_points_pre} -> {num_points_post}")

    if cloud.has_rgb:
        cloud.rgb = cloud.rgb[:, mask]


def apply_bounding_box_np(llbb: np.ndarray,
                          lubb: np.ndarray,
                          cloud: PointCloud,
                          bbox_extend: float = 0.1,
                          verbose: bool = False):
    mask = np.ones(cloud.xyz.shape[1], dtype=bool)
    for i in range(3):
        ax_mask = np.logical_and(cloud.xyz[i, :] > (llbb[i] - bbox_extend), cloud.xyz[i, :]
                                 < (lubb[i] + bbox_extend))
        mask = np.logical_and(ax_mask, mask)

    num_points_pre = cloud.xyz.shape[1]
    cloud.xyz = cloud.xyz[:, mask]
    num_points_post = cloud.xyz.shape[1]
    if verbose:
        print(f"Filtered from {num_points_pre} -> {num_points_post}")

    if cloud.has_rgb:
        cloud.rgb = cloud.rgb[:, mask]


def bbox_filter(Y: torch.Tensor, cloud: PointCloud, bbox_extend: float = 0.1):
    llbb, lubb = get_bounding_box(Y)
    apply_bounding_box(llbb, lubb, cloud, bbox_extend=bbox_extend)


def bbox_filter_np(Y: np.ndarray, cloud: np.ndarray, bbox_extend: float = 0.1):
    llbb, lubb = get_bounding_box_np(Y)
    apply_bounding_box_np(llbb, lubb, cloud, bbox_extend=bbox_extend)
