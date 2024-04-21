from pathlib import Path
from typing import List
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

FILE_DIR = Path(__file__).resolve().parent
REPO_DIR = FILE_DIR.parents[1]
sys.path.append(REPO_DIR.as_posix())

from limap_extension.constants import ImageDirection
from limap_extension.projection_based_flow import (projection_based_motion_segmentation,
                                                   flow_xyz_from_decomposed_motion,
                                                   segment_flow_xyz)
from limap_extension.img_cloud_transforms import reproject_img
from limap_extension.utils.io import (read_rgbd, read_pose, save_mask, flow_2d_to_rgb_and_mask,
                                      save_rgb, flow_mag_angle_to_rgb, scale_flow_viz)
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array
from limap_extension.optical_flow import OpticalFlow, Args, RAFT_MODEL_PATH


class FigureGenerator:

    def __init__(self, scenario: str, difficulty: str, trial: str, img_direction: ImageDirection,
                 frame_1_idx: int, frame_2_idx: int, output_dir: Path):
        self.scenario = scenario
        self.difficulty = difficulty
        self.trial = trial
        self.trial_path = REPO_DIR / "datasets" / scenario / difficulty / trial
        self.img_direction = img_direction
        self.frame_1_idx = frame_1_idx
        self.frame_2_idx = frame_2_idx
        self.output_dir = output_dir

        self.rgb_2_frame_2_path = self.output_dir / "rgb_2_frame_2.png"
        self.rgb_1_frame_2_path = self.output_dir / "rgb_1_frame_2.png"
        self.mask_valid_projection_cropped_path = self.output_dir / "mask_valid_projection_cropped.png"
        # the 4 panels of the image.
        self.flow_2d_rgb_path: Path = self.output_dir / "flow_2d_rgb.png"
        self.flow_2d_rgb_mask_path: Path = self.output_dir / "flow_2d_rgb_mask.png"
        self.flow_proj_rgb_path: Path = self.output_dir / "flow_proj_rgb.png"
        self.flow_proj_rgb_mask_path: Path = self.output_dir / "flow_proj_rgb_mask.png"

    def read_data(self):
        # Read data
        (rgb_2, depth_2) = read_rgbd(self.trial_path, self.img_direction, self.frame_2_idx)
        if self.frame_1_idx == 0:
            raise ValueError("Frame 1 index cannot be 0.")

        (rgb_1, depth_1) = read_rgbd(self.trial_path, self.img_direction, self.frame_1_idx)

        poses = read_pose(self.trial_path, self.img_direction)
        pose_1 = get_transform_matrix_from_pose_array(poses[self.frame_1_idx, :])
        pose_2 = get_transform_matrix_from_pose_array(poses[self.frame_2_idx, :])
        return rgb_1, depth_1, rgb_2, depth_2, pose_1, pose_2

    def generate_figure(self):
        rgb_1, depth_1, rgb_2, depth_2, pose_1, pose_2 = self.read_data()

        flow = OpticalFlow(None)
        flow.load_model(RAFT_MODEL_PATH, Args())

        img_dims_orig = depth_1.shape

        # Reproject the image at time t to the image frame at time t+1
        img_1_in_frame_2, depth_1_in_frame_2, mask_valid_projection, valid_bbox = reproject_img(
            rgb_1, depth_1, pose_1, pose_2)

        print("Save reprojection and original images??")

        img_1_in_frame_2_cropped = valid_bbox.crop_img(img_1_in_frame_2)
        depth_1_in_frame_2_cropped = valid_bbox.crop_img(depth_1_in_frame_2)
        rgb_2_cropped = valid_bbox.crop_img(rgb_2)
        depth_2_cropped = valid_bbox.crop_img(depth_2)
        mask_valid_projection_cropped = valid_bbox.crop_img(mask_valid_projection)

        save_mask(mask_valid_projection_cropped, self.mask_valid_projection_cropped_path)
        save_rgb(rgb_2_cropped, self.rgb_2_frame_2_path)
        save_rgb(img_1_in_frame_2_cropped, self.rgb_1_frame_2_path)

        # TODO: Make this a callable input to the function that either actually calculates the flow or
        # loads in the ground truth flow?
        # Might be easier to make this a flag and conditional.
        # TODO: For ground truth flow, we'll also need to project that from frame 1 into frame 2
        _, flow_up = flow.infer_flow(img_1_in_frame_2_cropped, rgb_2_cropped)
        flow_2d_rgb = flow.visualize(flow_up)
        flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()

        _, mask_2d_rgb = flow_2d_to_rgb_and_mask(flow_up)
        save_rgb(flow_2d_rgb.astype(np.uint8), self.flow_2d_rgb_path)
        save_mask(mask_2d_rgb.astype(np.uint8), self.flow_2d_rgb_mask_path)

        flow_reproj_rgb, mask_proj = self.generate_proj_flow_imgs(flow, flow_up,
                                                                  depth_1_in_frame_2_cropped,
                                                                  depth_2_cropped,
                                                                  mask_valid_projection_cropped,
                                                                  valid_bbox, img_dims_orig)

        save_rgb(flow_reproj_rgb.astype(np.uint8), self.flow_proj_rgb_path)
        save_mask(mask_proj.astype(np.uint8), self.flow_proj_rgb_mask_path)

    def generate_flow_2d_imgs(flow: 'OpticalFlow', flow_up: np.ndarray,
                              mask_valid_projection_cropped: np.ndarray):
        flow_2d_viz = flow_up.copy()
        flow_2d_viz[~mask_valid_projection_cropped] = 0

        flow_2d_rgb = flow.visualize(torch.from_numpy(flow_2d_viz).permute(2, 0, 1).unsqueeze(0))
        return scale_flow_viz(flow_2d_rgb)

    def generate_proj_flow_imgs(self, flow: OpticalFlow, flow_up: np.ndarray,
                                depth_1_in_frame_2_cropped: np.ndarray, depth_2_cropped: np.ndarray,
                                mask_valid_projection_cropped: np.ndarray, valid_bbox: np.ndarray,
                                img_dims_orig: np.ndarray):

        flow_xyz = flow_xyz_from_decomposed_motion(flow_up, depth_1_in_frame_2_cropped,
                                                   depth_2_cropped, mask_valid_projection_cropped)

        # mask_proj = segment_flow_xyz(flow_xyz)
        # mask_proj = valid_bbox.uncrop_img(mask_proj, *img_dims_orig, fill_value=False)

        # # Convert flow XYZ to RGB
        # # angle = np.dot(flow_xyz, np.array([1, 0, 0])) / np.linalg.norm(flow_xyz, axis=-1)
        # # angle = np.arccos(angle) / np.pi * 180.
        # # mag = np.linalg.norm(flow_xyz, axis=-1)
        # # flow_proj_rgb = flow_mag_angle_to_rgb(mag, angle)

        # # Need to convert the flow from 2D to 3D but while preserving the total magnitude. Should I
        # # use cross product?
        # # Have to be careful with coordinates here.

        # return flow_proj_rgb, mask_proj
        # flow_xyz_expand = valid_bbox.uncrop_img(flow_xyz, *img_dims_orig, fill_value=0)
        # flow_xyz_mag_tot = np.linalg.norm(flow_xyz_expand, axis=-1)
        flow_xyz_mag_tot = np.linalg.norm(flow_xyz, axis=-1)
        flow_xyz_mag_planar = np.linalg.norm(flow_xyz_expand[..., 1:], axis=-1)
        flow_xyz_mag_planar[flow_xyz_mag_planar < 1e-1] = 1e-1

        plt.figure()
        plt.imshow(flow_xyz_mag_tot)
        plt.figure()
        plt.imshow(flow_xyz_mag_planar)

        flow_scale = flow_xyz_mag_tot / flow_xyz_mag_planar

        flow_scale[~mask_valid_projection] = 0

        # flow_planar = valid_bbox.uncrop_img(flow_up.copy(), *rgb_1.shape[:-1], 0)
        # flow_planar_mag = np.linalg.norm(flow_planar, axis=-1)
        # mask_expand = mask_valid_projection
        # flow_planar_mag[~mask_expand] = 1.0
        # flow_planar_mag[flow_planar_mag < 1e-6] = 1e-6
        # flow_planar_mag_normed = flow_planar / flow_planar_mag[..., None]

        # flow_planar = np.round(flow_planar_mag_normed * flow_xyz_mag[..., None])
        # flow_planar = flow_planar * flow_mag[..., None]
        # Basically, we scale the flow_2d by the magnitude of the scale 3D (projection, but respects
        # magnitude)

        flow_up_expand = valid_bbox.uncrop_img(flow_up, *img_dims_orig, fill_value=0)
        flow_proj_vis = np.round(flow_scale[..., None] * flow_up_expand)

        flow_reproj_rgb = flow.visualize(
            torch.from_numpy(flow_proj_vis).permute(2, 0, 1).unsqueeze(0))

        plt.imshow(scale_flow_viz(flow_reproj_rgb))

    # def _make_figure(self, rgb_1: np.ndarray, depth_1: np.ndarray, rgb_2: np.ndarray,
    #                  depth_2: np.ndarray, pose_1: np.ndarray, flow: OpticalFlow):

    def get_target_files(self) -> List[Path]:
        return [
            self.flow_2d_rgb_path, self.flow_2d_rgb_mask_path, self.flow_proj_rgb_path,
            self.flow_proj_rgb_mask_path, self.rgb_2_frame_2_path, self.rgb_1_frame_2_path,
            self.get_figure_path()
        ]

    def get_figure_path(self) -> Path:
        return self.output_dir / "flow_figure.png"
