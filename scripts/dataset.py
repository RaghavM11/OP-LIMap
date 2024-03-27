import numpy as np
import os
import torch
from PIL import Image


DATASET = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P003'
class Dataset:

    def __init__(self, data_dir, shuffle =False):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.data_dict = {}  # dictionary to store the data
        self.load_data()
        
    
    def load_data(self):
        temp_dict = {}
        for name in os.listdir(self.data_dir):
            file_path = self.data_dir + '/' + name
            if os.path.isdir(file_path) and name != 'flow':
                temp_dict[name] =[]
                for file in os.listdir(file_path):
                    # check if the file is a .npy file or not
                    if file.endswith('.npy'):
                        # load the file
                        data = np.load(os.path.join(file_path, file))
                        # store the data in the dictionary
                        temp_dict[name].append(data)
                    else: #now for the images
                        # load the file
                        image = Image.open(os.path.join(file_path, file))
                        # store the data in the dictionary
                        # image_array = np.array(image)
                        temp_dict[name].append(image)
            elif name.endswith('.txt'):
                # load the file
                with open(os.path.join(file_path), 'r') as f:
                    pose = f.read()
                    # store the data in the dictionary
                    temp_dict[name] = data
        
        left_dict = {} 
        right_dict = {}
        # print(temp_dict.keys())
        for k in temp_dict.keys(): 
            if 'left' in k:
                loc = k.find('_')
                left_dict[k[:loc]] = np.array(temp_dict[k])
            elif 'right' in k:
                loc = k.find('_')
                right_dict[k[:loc]] = np.array(temp_dict[k])
            else:
                # print(temp_dict[k].shape)
                self.data_dict[k] = np.array(temp_dict[k])
        self.data_dict['left'] = left_dict
        self.data_dict['right'] = right_dict
        print(self.data_dict['right'].keys())
        self.N = self.data_dict['left']['image'].shape[0]
        self.H = self.data_dict['left']['image'].shape[1]
        self.W = self.data_dict['left']['image'].shape[2]
        print(self.H, self.W)
        print(self.data_dict['left'].keys(), self.data_dict.keys())
    
    def vectorise_data_image(self):
        self.data_dict['left']['image'] = self.data_dict['left']['image'].reshape((self.N, self.H *self.W, -1))
        self.data_dict['right']['image'] = self.data_dict['right']['image'].reshape((self.N, self.H *self.W, -1))
        print(self.data_dict['left']['image'].shape) 


if __name__ == '__main__':
    data_dir = DATASET
    dataset = Dataset(data_dir)
    print(dataset.data_dict.keys())
    dataset.vectorise_data_image()
    