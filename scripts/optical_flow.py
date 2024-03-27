# Optical Flow determination using Lucas-Kanade method

import cv2
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

class OpticalFlow:

    #takes the entire dataset class as the input
    def __init__(self, data, shuffle = False):
        self.data = data
        self.shuffle = shuffle
        self.flow_dict = {}
    
    def compute_sparse_flow(self):
        data_dict = self.data.data_dict
        N = self.data.N

        for i in range(N-1):
            l_t = data_dict['left']['image'][i]
            r_t = data_dict['right']['image'][i]
            l_t1 = data_dict['left']['image'][i+1]
            r_t1 = data_dict['right']['image'][i+1]
    
    def compute_dense_flow(self):
        pass




        
