import sys
from pathlib import Path

import numpy as np

from PIL import Image
import cv2

REPO_DIR = Path(".").resolve().parents[0]
# print(REPO_DIR)
sys.path.append(REPO_DIR.as_posix())

from limap_extension.optical_flow import Args, OpticalFlow, RAFT_MODEL_PATH
from limap_extension.img_cloud_transforms import reproject_img, uvz_ned_to_xyz_cam, get_uv_coords, index_img_with_uv_coords, xyz_cam_to_uvz_ned
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array


def preprocess_valid_projection_mask(mask_valid_proj: np.ndarray):
    # Do some processing to invalidate the areas around the projection mask. This deals with
    # repeated, thin, sharp objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_adjusted = cv2.morphologyEx(mask_valid_proj.astype(np.uint8),
                                     cv2.MORPH_OPEN,
                                     kernel,
                                     iterations=5).astype(bool)
    return mask_adjusted


def flow_xyz_from_decomposed_motion(flow_up: np.ndarray,
                                    depth_1_cropped: np.ndarray,
                                    depth_2_cropped: np.ndarray,
                                    mask_valid_projection_cropped: np.ndarray,
                                    is_returning_associated_uvs_orig: bool = False):
    # Idea: decompose the flow field into planar and depth components.
    proj_mask_to_use = preprocess_valid_projection_mask(mask_valid_projection_cropped)

    d1c_valid = depth_1_cropped[proj_mask_to_use]
    d2c_valid = depth_2_cropped[proj_mask_to_use]

    us, vs = get_uv_coords(*flow_up.shape[:-1])
    us = us[proj_mask_to_use.flatten()]
    vs = vs[proj_mask_to_use.flatten()]

    dus = flow_up[proj_mask_to_use, 1].flatten()
    dvs = flow_up[proj_mask_to_use, 0].flatten()

    new_us_float = us + dus
    new_vs_float = vs + dvs

    zs_1_in_frame_2 = d1c_valid.reshape(-1)

    GRABBING_FROM_DEPTH_2 = True
    if GRABBING_FROM_DEPTH_2:
        zs_2 = d2c_valid

        # Additional consideration for indices outside of the frame.
        d2_rows, d2_cols = depth_2_cropped.shape

        new_us_int = np.round(new_us_float).astype(int)
        new_vs_int = np.round(new_vs_float).astype(int)
        valid_idxs = (new_us_int >= 0) & (new_us_int < d2_cols) & (new_vs_int >= 0) & (new_vs_int
                                                                                       < d2_rows)
        us = us[valid_idxs]
        vs = vs[valid_idxs]
        dus = dus[valid_idxs]
        dvs = dvs[valid_idxs]
        new_us_float = us + dus
        new_vs_float = vs + dvs
        new_us_int = np.round(new_us_float).astype(int)
        new_vs_int = np.round(new_vs_float).astype(int)
        zs_1_in_frame_2 = zs_1_in_frame_2[valid_idxs]
        zs_2 = index_img_with_uv_coords(new_us_int, new_vs_int, depth_2_cropped)
    else:
        zs_2 = zs_1_in_frame_2

    # TODO: We should be able to accomplish this just with matrix multiplication.
    # i.e. H_NED_TO_CAM @ zs_1_in_frame_2 @ K_inv @ [dus, dvs, 1]
    # That's more pseudocode than actual code since it ignores dimensionality issues.
    xyz_1 = uvz_ned_to_xyz_cam(us, vs, zs_1_in_frame_2)

    # Trying this out.
    xyz_2 = uvz_ned_to_xyz_cam(new_us_float, new_vs_float, zs_2)
    # zs_2_orig = depth_2_cropped.reshape(-1)
    # zs_2 = depth_2_cropped.reshape(-1)
    # zs_2_shifted = index_img_with_uv_coords(us + dus, vs + dvs, depth_2_cropped)
    # xyz_2 = uvz_ned_to_xyz_cam(us + dus, vs + dvs, zs_2_shifted)

    if GRABBING_FROM_DEPTH_2:
        # planar_motion = np.zeros((depth_2_cropped.size, 2))
        flow_xyz = np.zeros((*flow_up.shape[:-1], 3))
        # Since there's a one-to-one correspondence between the u/vs here and the delta x/ys in
        # xyz_2, we project these values back into image 2.
        us, vs, _ = xyz_cam_to_uvz_ned(xyz_2, is_rounding_to_int=True)
        flow_xyz[vs, us, :] = xyz_2 - xyz_1
    else:
        delta_xy = (xyz_1 - xyz_2)[:, :-1]
        planar_motion = delta_xy.reshape(*flow_up.shape)

        x_ned_motion = planar_motion[:, :, 0]
        y_ned_motion = planar_motion[:, :, 1]

        # Now, depth distance can be calculated by the difference in depth between the two frames.
        depth_motion = np.abs(depth_2_cropped - depth_1_cropped)

        # New idea, just consider planar motion since z-based motion is wrong (should be indexing
        # image frame 2 depths with shifted coordinates).
        # depth_motion = np.zeros_like(depth_2_cropped)

        flow_xyz = np.stack((x_ned_motion, y_ned_motion, depth_motion), axis=-1)
        flow_xyz[~proj_mask_to_use] = 0.0

    return flow_xyz


def flow_xyz_planar_motion_only(flow_up: np.ndarray, depth_1_cropped: np.ndarray,
                                depth_2_cropped: np.ndarray, mask_valid_projection_cropped):
    # Idea: decompose the flow field into planar and depth components.
    proj_mask_to_use = preprocess_valid_projection_mask(mask_valid_projection_cropped)
    us, vs = get_uv_coords(*flow_up.shape[:-1])
    dus = flow_up[:, :, 1].flatten()
    dvs = flow_up[:, :, 0].flatten()
    zs_1_in_frame_2 = depth_1_cropped.reshape(-1)

    # TODO: We should be able to accomplish this just with matrix multiplication.
    # i.e. H_NED_TO_CAM @ zs_1_in_frame_2 @ K_inv @ [dus, dvs, 1]
    # That's more pseudocode than actual code since it ignores dimensionality issues.
    xyz_1 = uvz_ned_to_xyz_cam(us, vs, zs_1_in_frame_2)

    # Trying this out.
    zs_2 = zs_1_in_frame_2
    xyz_2 = uvz_ned_to_xyz_cam(us + dus, vs + dvs, zs_2)
    # zs_2_orig = depth_2_cropped.reshape(-1)
    # zs_2 = depth_2_cropped.reshape(-1)
    # zs_2_shifted = index_img_with_uv_coords(us + dus, vs + dvs, depth_2_cropped)
    # xyz_2 = uvz_ned_to_xyz_cam(us + dus, vs + dvs, zs_2_shifted)

    delta_xy = (xyz_1 - xyz_2)[:, :-1]
    planar_motion = delta_xy.reshape(*flow_up.shape)

    x_ned_motion = planar_motion[:, :, 0]
    y_ned_motion = planar_motion[:, :, 1]

    # New idea, just consider planar motion since z-based motion is wrong (should be indexing
    # image frame 2 depths with shifted coordinates).
    depth_motion = np.zeros_like(depth_2_cropped)

    flow_planar = np.stack((x_ned_motion, y_ned_motion, depth_motion), axis=-1)
    flow_planar[~proj_mask_to_use] = 0.0

    return flow_planar


def segment_flow_xyz(flow_xyz: np.ndarray, threshold: float = 0.4):
    print("WARNING: Flow segmentation should use a *velocity* threshold, not positional.")
    flow_mag = np.linalg.norm(flow_xyz, axis=-1)
    flow_mask = flow_mag > threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    flow_mask_adjusted = flow_mask.astype(np.uint8)
    flow_mask_adjusted = cv2.erode(flow_mask_adjusted, kernel)
    flow_mask_adjusted = cv2.morphologyEx(flow_mask_adjusted, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    flow_mask_adjusted = cv2.dilate(flow_mask_adjusted, kernel)

    return flow_mask_adjusted.astype(bool)


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

    # TODO: Make this a callable input to the function that either actually calculates the flow or
    # loads in the ground truth flow?
    # Might be easier to make this a flag and conditional.
    # TODO: For ground truth flow, we'll also need to project that from frame 1 into frame 2
    _, flow_up = flow.infer_flow(img_1_in_frame_2_cropped, rgb_2_cropped)
    flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()

    flow_xyz = flow_xyz_from_decomposed_motion(flow_up, depth_1_in_frame_2_cropped, depth_2_cropped,
                                               mask_valid_projection_cropped)

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
