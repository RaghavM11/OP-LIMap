# Optical Flow determination using Lucas-Kanade method

import os

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation

from limap_extension.raft.raft import RAFT
from limap_extension.utils import flow_viz
from limap_extension.utils.utils import InputPadder
from limap_extension.img_cloud_transforms import reproject_img
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array


class Args():

    def __init__(self, model='', path='', small=False, mixed_precision=True, alternate_corr=False):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class OpticalFlow:

    #takes the entire dataset class as the input
    def __init__(self, data, shuffle=False):
        self.data = data
        self.shuffle = shuffle
        self.flow_dict = {}

    def load_model(self, path, args):
        self.model = RAFT(args)
        weights = torch.load(path, map_location='cpu')
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(weights)
        self.model.to('cpu')

    def infer_flow(self, fr1, fr2):
        self.model.eval()
        with torch.no_grad():
            fr1 = torch.from_numpy(fr1).permute(2, 0, 1).float()[None]
            fr2 = torch.from_numpy(fr2).permute(2, 0, 1).float()[None]

            padder = InputPadder(fr1.shape, mode='sintel')
            fr1, fr2 = padder.pad(fr1, fr2)
            flow_low, flow_up = self.model(fr1,
                                           fr2,
                                           iters=20,
                                           upsample=True,
                                           flow_init=None,
                                           test_mode=True)
            return flow_low, flow_up

    def visualize(self, flow):
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        flow = flow_viz.flow_to_image(flow)
        return flow


'''
Check for the motion segmentation and segmenting static adn dynamic objects:
1) Project the frame at time t to the frame at time t+1
2) Compute the optical flow between the two frames
3) Compute the flow magnitude
4) Threshold the flow magnitude to get the static and dynamic objects
5) Segment the objects based on the threshold

Arguments: frame1, frame2, camera_pose1, camera_pose2
'''


def motion_segmentation(rgb_1: np.ndarray, depth_1: np.ndarray, rgb_2: np.ndarray,
                        depth_2: np.ndarray, cp1: np.ndarray, cp2: np.ndarray):
    flow = OpticalFlow(None)
    path = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/raft-sintel.pth'
    flow.load_model(path, Args())

    pose_1 = get_transform_matrix_from_pose_array(cp1)
    pose_2 = get_transform_matrix_from_pose_array(cp2)

    # Reproject the image at time t to the image frame at time t+1
    img_1_in_frame_2, valid_bbox = reproject_img(rgb_1, depth_1, pose_1, pose_2)

    # Crop the images to the valid bounding box
    img_1_in_frame_2 = valid_bbox.crop_img(img_1_in_frame_2)
    rgb_2 = valid_bbox.crop_img(rgb_2)

    # Compute the optical flow between the two images in the same frame to determine dynamic
    # objects.
    flow_low, flow_up = flow.infer_flow(rgb_2, img_1_in_frame_2)
    # print(flow_up, flow_low)
    flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
    mask = np.zeros_like(rgb_1)
    mask[..., 1] = 255
    mag, angle = cv2.cartToPolar(flow_up[..., 0], flow_up[..., 1], angleInDegrees=True)

    mask[:, :, 0] = 0
    mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
    mask_grey = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    mean = np.mean(mask_grey)
    print(mean)
    std_deviation = np.std(mask_grey)
    print(std_deviation)
    mask_grey = (mask_grey - mean) / std_deviation
    print(np.mean(mask_grey))
    mask_grey = np.where(mask_grey < 0, 0, mask_grey)
    mask_grey = np.where(mask_grey > 0.5, 1, 0)
    mask_r = np.mean(mask_rgb[:, :, 0])
    mask_g = np.mean(mask_rgb[:, :, 1])
    mask_b = np.mean(mask_rgb[:, :, 2])
    cov_mask_r = np.std(mask_rgb[:, :, 0])
    cov_mask_g = np.std(mask_rgb[:, :, 1])
    cov_mask_b = np.std(mask_rgb[:, :, 2])
    mask_r = (mask_rgb[:, :, 0] - mask_r) / cov_mask_r
    mask_g = (mask_rgb[:, :, 1] - mask_g) / cov_mask_g
    mask_b = (mask_rgb[:, :, 2] - mask_b) / cov_mask_b
    mask_rgb = np.stack([mask_r, mask_g, mask_b], axis=2)

    # do binary dilation on the grey_mask
    mask_grey = binary_dilation(mask_grey, iterations=5)

    # mask = np.asarray(mask_rgb[..., 2], dtype=np.uint8)
    # mask[mask < threshold] =
    # mask[mask >= threshold] = 1

    # Save it

    # plt.figure()
    # plt.imshow(mask_grey, cmap='gray')
    # plt.show()
    return mask_rgb, mask_grey


# NOTE: Moved the runner code to scripts/run_flow_example.py
