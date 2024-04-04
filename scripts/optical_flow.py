# Optical Flow determination using Lucas-Kanade method

import cv2
import numpy as np
import os
import torch
from PIL import Image
from raft import RAFT
import matplotlib.pyplot as plt
from utils import flow_viz
from utils.utils import InputPadder


class Args():
  def __init__(self, model='', path='', small=False, 
               mixed_precision=True, alternate_corr=False):
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
    def __init__(self, data, shuffle = False):
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

            padder = InputPadder(fr1.shape, mode = 'sintel')
            fr1, fr2 = padder.pad(fr1, fr2)
            flow_low, flow_up = self.model(fr1, fr2, iters=20, upsample=True, flow_init = None, test_mode=True)
            return flow_low, flow_up
    
    def visualize(self, flow):
        flow = flow[0].permute(1,2,0).cpu().numpy()
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
def motion_segmentation(fr1, fr2, depth1, depth2, cp1, cp2):
   from scipy.spatial.transform import Rotation as R
   flow = OpticalFlow(None)
   path = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/raft-sintel.pth'
   flow.load_model(path, Args())
   # Now make the poses into the 4x4 matrices
   pose1 = np.eye(4)
   pose2 = np.eye(4)
   # pcd1 
   # pcd2 

   # Now compute the transformation from pose1 to pose2
   pose1[:3, :3] = R.from_quat(cp1[3:]).as_matrix()
   pose1[:3, 3] = cp1[:3] 
   pose2[:3, :3] = R.from_quat(cp2[3:]).as_matrix()
   pose2[:3, 3] = cp2[:3]
   p1_to_p2 = np.linalg.inv(pose2) @ pose1

   # given the transformation matrix, we can now project the points from the first frame to the second frame
   # transformed pcd inot the new camera frame
   # project it back to the image plane

    # Now we can compute the optical flow between the two frames
    #flow_low, flow_up = flow.infer_flow(projected, fr2)
    #flow_mag = np.linalg.norm(flow_up, axis = 0)
    #threshold = 0.5
    #static = flow_mag < threshold
    #dynamic = flow_mag >= threshold
    #return static, dynamic





   



if __name__ == '__main__':
    ten = torch.tensor([1,2,3,4,5])
    print(ten.device)
    path = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/raft-sintel.pth'
    flow = OpticalFlow(None)
    flow.load_model(path, Args())
    fr1 = np.array(Image.open('/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P006/image_left/000091_left.png'))
    fr2 = np.array(Image.open('/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P006/image_left/000092_left.png'))
    flow_low, flow_up = flow.infer_flow(fr1, fr2)
    flow_viz = flow.visualize(flow_up)
    flow_viz = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)
    flow_viz = np.concatenate([fr1, flow_viz], axis = 0)
    cv2.imshow('flow', flow_viz/255.0)
    cv2.waitKey()
    