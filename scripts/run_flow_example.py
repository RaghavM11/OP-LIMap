import torch
import numpy as np
from PIL import Image

from limap_extension.optical_flow import OpticalFlow, Args, motion_segmentation

if __name__ == '__main__':
    ten = torch.tensor([1, 2, 3, 4, 5])
    print(ten.device)
    path = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/raft-sintel.pth'
    flow = OpticalFlow(None)
    flow.load_model(path, Args())
    fr1 = np.array(
        Image.open(
            '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P006/image_left/000091_left.png'
        ))
    fr2 = np.array(
        Image.open(
            '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P006/image_left/000092_left.png'
        ))
    depth1 = np.load(
        '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P006/depth_left/000091_left_depth.npy'
    )
    depth2 = np.load(
        '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P006/depth_left/000092_left_depth.npy'
    )

    cam_pose = np.loadtxt(
        '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P006/pose_left.txt'
    )
    # print(cam_pose.shape)
    flow_low, flow_up = flow.infer_flow(fr1, fr2)
    # flow_viz = flow.visualize(flow_up)
    # flow_viz = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)
    # flow_viz = np.concatenate([fr1, flow_viz], axis=0)
    motion_segmentation(fr1, depth1, fr2, depth2, cam_pose[91, :], cam_pose[92, :])
    # cv2.imshow('flow', flow_viz / 255.0)
    # cv2.waitKey()
