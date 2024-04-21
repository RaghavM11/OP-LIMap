from typing import Tuple, Optional, List, Dict
import copy

import numpy as np
import torch
# import cv2

from limap_extension.constants import CAM_INTRINSIC, H_OCV_TO_NED, H_NED_TO_OCV
from limap_extension.bounding_box import BoundingBox, CornerIdxs
from limap_extension.point_cloud import PointCloud
import limap_extension.transformation_tartanair as tf
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array

# TartainAir depth unit is meters.
MILLIMETERS_TO_METERS = 1e-3
METERS_TO_METERS = 1.0


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    R = pose[:3, :3]
    t = pose[:3, 3]
    inv_R = R.T
    inv_t = -inv_R @ t
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = inv_R
    inv_pose[:3, 3] = inv_t
    return inv_pose


def ned2cam_single_pose(pose: np.ndarray) -> np.ndarray:
    """transfer a ned traj to camera frame traj

    INPUT TO THIS FUNCTION SHOULD BE IN NED FRAME
    """
    T = H_NED_TO_OCV
    pose_mat = get_transform_matrix_from_pose_array(pose)
    new_pose = T @ pose_mat @ inverse_pose(T)

    return new_pose


def cam2ned_single_pose(pose: np.ndarray) -> np.ndarray:
    """transfer a camera traj to ned frame traj

    INPUT TO THIS FUNCTION SHOULD BE IN CAMERA/WORLD FRAME
    """
    T = H_OCV_TO_NED
    pose_mat = get_transform_matrix_from_pose_array(pose)
    new_pose = T @ pose_mat @ inverse_pose(T)

    return new_pose


def index_img_with_uv_coords(us: np.ndarray, vs: np.ndarray, img: np.ndarray) -> np.ndarray:
    return img[vs, us, ...]


def get_uv_coords(img_rows, img_cols):
    # U, V = np.meshgrid(np.arange(img_rows), np.arange(img_cols))
    V, U = np.meshgrid(np.arange(img_rows), np.arange(img_cols))
    # U, V = np.meshgrid(np.arange(img_cols), np.arange(img_rows))
    # V, U = np.meshgrid(np.arange(img_cols), np.arange(img_rows))
    # print("U shape before flatten:", U.shape)
    V = V.T.flatten()
    U = U.T.flatten()
    # print("U max:", np.max(U), "V max:", np.max(V))
    return U, V


def augment_coords(coords: np.ndarray):
    if ((coords.shape[1] != 3) and (coords.shape[1] != 2)):
        raise ValueError(f"Coords must have 2 or 3 columns, got shape {coords.shape}")
    return np.hstack([coords, np.ones((coords.shape[0], 1))])


def tform_coords(tform: np.ndarray, coords: np.ndarray) -> np.ndarray:
    coords_aug = augment_coords(coords)
    return (tform @ coords_aug.T).T[:, :-1]


def find_valid_uv_coords(us: np.ndarray, vs: np.ndarray, img_height: int,
                         img_width: int) -> np.ndarray:
    """Returns mask of where the UV coordinates are valid given image dimensions"""
    if us.shape != vs.shape:
        raise ValueError("Us and Vs must have the same shape.")

    if us.dtype != int or vs.dtype != int:
        raise ValueError("Us and Vs must be integers.")

    where_u_valid = np.logical_and(us >= 0, us < img_width)
    where_v_valid = np.logical_and(vs >= 0, vs < img_height)
    where_uv_valid = np.logical_and(where_u_valid, where_v_valid)
    return where_uv_valid


def uvz_ocv_to_xyz_ned(us: np.ndarray,
                       vs: np.ndarray,
                       zs: np.ndarray,
                       intrinsic: np.ndarray = CAM_INTRINSIC,
                       depth_units_to_tracked_units: float = METERS_TO_METERS) -> np.ndarray:
    fu = intrinsic[0, 0]
    fv = intrinsic[1, 1]

    center_u = intrinsic[0, 2]
    center_v = intrinsic[1, 2]

    constant_u = 1.0 / fu
    constant_v = 1.0 / fv
    xs = (us - center_u) * zs * depth_units_to_tracked_units * constant_u
    ys = (vs - center_v) * zs * depth_units_to_tracked_units * constant_v
    zs = zs * depth_units_to_tracked_units

    # xyzs_img = np.stack((xs, ys, zs), axis=1)
    xyzs_ned = np.stack((xs, ys, zs), axis=1)
    # print("xyzs_img shape:", xyzs_ned.shape)
    # print("xyzs_ned maxes:", np.max(xyzs_ned, axis=0))
    # xyzs_cam = (H_IMG_TO_CAM @ np.hstack([xyzs_img, np.ones((xyzs_img.shape[0], 1))]).T).T[:, :-1]
    # xyzs_cam = tform_coords(np.linalg.inv(H_IMG_TO_CAM), xyzs_img)
    xyzs_cam = tform_coords(H_NED_TO_OCV, xyzs_ned)
    # xyzs_cam =
    return xyzs_cam
    # return xyzs_ned


def xyz_ned_to_uvz_ocv(
        xyz: np.ndarray,
        is_rounding_to_int: bool = False,
        intrinsic: np.ndarray = CAM_INTRINSIC,
        depth_units_to_tracked_units: float = METERS_TO_METERS) -> Tuple[np.ndarray, np.ndarray]:
    # xyz = tform_coords(H_IMG_TO_CAM, xyz)
    xyz = tform_coords(H_OCV_TO_NED, xyz)
    fu = intrinsic[0, 0]
    fv = intrinsic[1, 1]

    center_u = intrinsic[0, 2]
    center_v = intrinsic[1, 2]

    constant_u = 1.0 / fu
    constant_v = 1.0 / fv

    # xyz = xyz.T
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    zs = xyz[:, 2]
    zs = zs / depth_units_to_tracked_units

    us = (xs / (zs * depth_units_to_tracked_units * constant_u)) + center_u
    vs = (ys / (zs * depth_units_to_tracked_units * constant_v)) + center_v

    if is_rounding_to_int:
        print(
            "Need to check depth of the coordinates that correspond to the same pixel so that we can select the one with the smallest depth."
        )
        us = np.round(us).astype(int)
        vs = np.round(vs).astype(int)

    # uv_coords = np.round(np.stack((us, vs), axis=0)).astype(int)

    return us, vs, zs


def imgs_to_clouds_np(
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        intrinsic: np.ndarray,
        depth_units_to_tracked_units: float = METERS_TO_METERS) -> Tuple[PointCloud, CornerIdxs]:
    """Converts RGB/D images to point clouds represented by XYZ and RGB arrays

    This has a lot of commented code that can be used to filter out points given a 2D (segmentation)
    mask. I don't think we'll need this but I'm keeping it in case we do.
    """
    # fx = intrinsic[0, 0]
    # fy = intrinsic[1, 1]

    # center_x = intrinsic[0, 2]
    # center_y = intrinsic[1, 2]

    # constant_x = 1.0 / fx
    # constant_y = 1.0 / fy

    U, V = get_uv_coords(depth_img.shape[0], depth_img.shape[1])

    # print("U[:10]:", U[:10])
    # print("V[:10]:", V[:10])

    # Now we get the coordinates of the corners. This helps down the line to crop the view to only
    # the parts of the image that are visible in the other image.
    # Due to numpy flattening in row-major convention, corner coordinates are:
    # 0, img_rows - 1, img_rows * img_cols, img_rows * img_cols + img_cols
    # corner_idxs = {
    #     "upper_left": 0,
    #     # "upper_right": -depth_img.shape[1],
    #     # "lower_left": depth_img.shape[1] - 1,
    #     "upper_right": depth_img.shape[1] - 1,
    #     "lower_left": -depth_img.shape[1],
    #     "lower_right": -1
    # }
    # corner_idxs = CornerIdxs(0, -depth_img.shape[0], depth_img.shape[0] - 1, -1)
    # corner_idxs = CornerIdxs(0, depth_img.shape[0] - 1, -depth_img.shape[0], -1)
    corner_idxs = CornerIdxs(0, depth_img.shape[1] - 1, -depth_img.shape[1], -1)
    # corner_idxs = [0, -depth_img.shape[1], depth_img.shape[1] - 1, -1]
    # corner_idxs = [0, depth_img.shape[1] - 1, -depth_img.shape[1], -1]

    # print("U[corner_idxs.upper_left]", U[corner_idxs.upper_left])
    # print("V[corner_idxs.upper_left]", V[corner_idxs.upper_left])
    # print("U[corner_idxs.upper_right]", U[corner_idxs.upper_right])
    # print("V[corner_idxs.upper_right]", V[corner_idxs.upper_right])
    # print("U[corner_idxs.lower_left]", U[corner_idxs.lower_left])
    # print("V[corner_idxs.lower_left]", V[corner_idxs.lower_left])
    # print("U[corner_idxs.lower_right]", U[corner_idxs.lower_right])
    # print("V[corner_idxs.lower_right]", V[corner_idxs.lower_right])

    # Extremely heavy handed but I want to be sure I've got this right.
    assert U[corner_idxs.upper_left] == 0
    assert V[corner_idxs.upper_left] == 0
    assert U[corner_idxs.upper_right] == depth_img.shape[1] - 1
    assert V[corner_idxs.upper_right] == 0
    assert U[corner_idxs.lower_left] == 0
    assert V[corner_idxs.lower_left] == depth_img.shape[0] - 1
    assert U[corner_idxs.lower_right] == depth_img.shape[1] - 1
    assert V[corner_idxs.lower_right] == depth_img.shape[0] - 1
    # print("Upper left:", U[corner_idxs[0]], V[corner_idxs[0]])
    # print("Upper right:", U[corner_idxs[1]], V[corner_idxs[1]])
    # print("Lower left:", U[corner_idxs[2]], V[corner_idxs[2]])
    # print("Lower right:", U[corner_idxs[3]], V[corner_idxs[3]])

    depth_flat = depth_img.flatten()

    # Get individual values.
    # NOTE: x is the depth here!!
    xyz_cam = uvz_ocv_to_xyz_ned(U, V, depth_flat, intrinsic, depth_units_to_tracked_units)
    # print("xyz_cam mins:", np.min(xyz_cam, axis=0))
    # print("xyz_cam maxes:", np.max(xyz_cam, axis=0))
    xs, ys, zs = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]

    # Z = 0 indicates the point is invalid in the depth images that I've been working with in the
    # lab. I'm not sure if TartanAir has a similar convention.
    where_depth_valid = zs != 0.0

    num_invalid_depths = np.count_nonzero(~where_depth_valid)
    if num_invalid_depths > 0:
        raise ValueError(f"Found {num_invalid_depths} invalid depths in the depth image. Indicates "
                         "coordinate transform issue since z should be depth and not zero in "
                         "tartanair dataset.")

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

    corner_idxs.adjust_coords_given_depth_validity(where_depth_valid, depth_img.shape[0])

    # print("Corner indexes:", corner_idxs.as_np_array())

    return PointCloud(xyz_cloud_unfiltered.T, rgb_cloud_unfiltered.T), corner_idxs


def cloud_to_img_np(
    cloud: PointCloud,
    intrinsic: np.ndarray,
    img_width: int = 640,
    img_height: int = 480,
    depth_units_to_tracked_units: float = METERS_TO_METERS,
    corner_idxs: Optional[CornerIdxs] = None,
    interpolation_method: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, BoundingBox]:
    """Turns cloud coordinates to UV (image) coordinates

    TODO: Do we need to do some sort of bilinear interpolation here to correct for camera
    distortions? While working with images from real sensors, I've seen that reprojection from 3D to
    2D rasterized images can leave holes in the image due to camera distortions.
    """
    # fx = intrinsic[0, 0]
    # fy = intrinsic[1, 1]

    # center_x = intrinsic[0, 2]
    # center_y = intrinsic[1, 2]

    # constant_x = 1.0 / fx
    # constant_y = 1.0 / fy

    # xs = cloud.xyz[:, 0]
    # ys = cloud.xyz[:, 1]
    # zs = cloud.xyz[:, 2] / depth_units_to_tracked_units

    # vs = xs / (zs * depth_units_to_tracked_units * constant_x) + center_x
    # us = ys / (zs * depth_units_to_tracked_units * constant_y) + center_y

    us, vs, zs = xyz_ned_to_uvz_ocv(cloud.xyz,
                                    intrinsic=intrinsic,
                                    depth_units_to_tracked_units=depth_units_to_tracked_units)
    # print("us max:", np.max(us))
    # print("vs max:", np.max(vs))
    # print("zs max:", np.max(zs))

    # print("us min:", np.min(us))
    # print("vs min:", np.min(vs))
    # print("zs min:", np.min(zs))

    if corner_idxs is not None:
        # If no corner indexes are provided, the valid bounding box is the entire image.
        uv_coords = np.round(np.stack((us, vs), axis=0)).astype(int)
        valid_bbox = BoundingBox.from_cloud_corner_idxs(corner_idxs, uv_coords, img_height,
                                                        img_width)
    else:
        # If corner indexes are provided, we can use them to get a tighter bounding box on what
        # parts of image 1 are visible in image 2. This should help the flow out.
        valid_bbox = BoundingBox(0, img_width - 1, 0, img_height - 1)

    if interpolation_method is None:
        us = np.round(us).astype(int)
        vs = np.round(vs).astype(int)

    where_u_valid = np.logical_and(us >= 0, us < img_width)
    where_v_valid = np.logical_and(vs >= 0, vs < img_height)
    where_uv_valid = np.logical_and(where_u_valid, where_v_valid)

    us_valid = us[where_uv_valid]
    vs_valid = vs[where_uv_valid]
    # uv_coords = uv_coords[:, where_uv_valid]
    zs = zs[where_uv_valid]
    rgb_valid = cloud.rgb[where_uv_valid, :]

    # We could try to interpolate the non-rasterized points into the rasterized grid (I think numpy
    # has a function for this)
    if interpolation_method is None:
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        # print("uv_coords maxes:", np.max(uv_coords, axis=1))
        img[vs_valid, us_valid, :] = rgb_valid

        depth_img = np.zeros((img_height, img_width), dtype=np.float32)
        depth_img[vs_valid, us_valid] = zs

        mask_valid_projection = np.zeros((img_height, img_width), dtype=bool)
        mask_valid_projection[vs_valid, us_valid] = True
        from scipy.ndimage import binary_erosion
        kernel = np.ones((3, 3))
        mask_valid_projection = binary_erosion(mask_valid_projection, kernel)
    elif interpolation_method == "clough_tocher":
        raise NotImplementedError("Clough-Tocher interpolation not yet implemented.")
        from scipy.interpolate import CloughTocher2DInterpolator
        # from scipy.interpolate import LinearNDInterpolator
        uv_coords = uv_coords.T
        rgb_float = rgb_valid.astype(float)
        U, V = get_uv_coords(img_height, img_width)
        uv_coords_interp = np.stack((V.flatten(), U.flatten()), axis=1)
        # print(uv_coords_interp.shape)
        channels = []
        for i in range(3):
            interp = CloughTocher2DInterpolator(uv_coords, rgb_float[:, i])
            # interp = LinearNDInterpolator(uv_coords, rgb_float[:, i])
            channel = interp(uv_coords_interp)

            # Process the output
            channel = np.nan_to_num(channel, nan=0.0, posinf=255.0, neginf=0.0)
            channel[channel < 0] = 0
            channel[channel > 255] = 255

            channels.append(channel.reshape((img_height, img_width)).astype(np.uint8))
        img = np.stack(channels, axis=2).astype(np.uint8)
    elif interpolation_method == "interpn":
        # from scipy.interpolate import interpn
        raise NotImplementedError("Interpolation method 'interpn' not yet implemented.")
    else:
        raise ValueError(f"Interpolation method {interpolation_method} not recognized.")

    # Old for loop implementation. Keeping this in case we need it for the intepolation we might
    # wind up doing to handle camera distortions.
    # for i in range(uv_coords.shape[1]):
    #     v, u = uv_coords[:, i]
    #     if v >= img_width or u >= img_height:
    #         print(f"({u}, {v}) out of bounds!")
    #         continue
    #     rgb = cloud.rgb[i, :]
    #     img[u, v, :] = rgb

    return img, depth_img, mask_valid_projection, valid_bbox


def transform_cloud(cloud: PointCloud, H: np.ndarray) -> PointCloud:
    """Transforms a point coud from one frame to another using a homogeneous tform matrix"""
    cloud_tformed = copy.deepcopy(cloud)

    # Augment the xyz cloud with 1s so transform matmul works.
    # xyz_aug = np.hstack([cloud_tformed.xyz, np.ones((cloud_tformed.xyz.shape[0], 1))])

    # xyz_tformed = (H @ xyz_aug.T).T
    cloud_tformed.xyz = tform_coords(H, cloud_tformed.xyz)
    return cloud_tformed


def reproject_img(
    rgb_1: np.ndarray,
    depth_1: np.ndarray,
    pose_1_world: np.ndarray,
    pose_2_world: np.ndarray,
    interpolation_method: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, BoundingBox]:
    """Reprojects an image from frame 1 to frame 2 using the poses of the two frames."""
    # TODO: The H_OCV_TO_NED left-multiplication is unnecessary as the poses are already in world
    # frame (not NED).
    T_world_to_cam_1 = pose_1_world
    T_world_to_cam_2 = pose_2_world
    # This is the direction of forward movement in time.
    T_cam_1_to_cam_2 = inverse_pose(T_world_to_cam_1) @ T_world_to_cam_2

    # Cloud should be in NED (z depth) frame.
    cloud_frame_1, corner_idxs = imgs_to_clouds_np(rgb_1, depth_1, CAM_INTRINSIC)
    # print("Cloud frame 1 maxes:", np.max(cloud_frame_1.xyz, axis=0))
    # print("Cloud frame 1 mins:", np.min(cloud_frame_1.xyz, axis=0))

    cloud_tformed = transform_cloud(cloud_frame_1, H_OCV_TO_NED @ T_cam_1_to_cam_2 @ H_NED_TO_OCV)
    # print("Transformed Cloud maxes:", np.max(cloud_tformed.xyz, axis=0))
    # print("Transformed Cloud mins:", np.min(cloud_tformed.xyz, axis=0))
    img_tformed, depth_tformed, mask_valid_projection, valid_bbox = cloud_to_img_np(
        cloud_tformed,
        CAM_INTRINSIC,
        corner_idxs=corner_idxs,
        interpolation_method=interpolation_method)
    return img_tformed, depth_tformed, mask_valid_projection, valid_bbox


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
