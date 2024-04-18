import os, sys
import numpy as np
import cv2
import h5py#chnage file format

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base

#check function for file format

def read_raydepth(raydepth_fname, resize_hw=None, max_image_dim=None):
    with h5py.File(raydepth_fname, 'r') as f:
        raydepth = np.array(f['dataset']).astype(np.float32)
    if resize_hw is not None and raydepth.shape != resize_hw:
        raydepth = cv2.resize(raydepth, (resize_hw[1], resize_hw[0]))
    if (max_image_dim is not None) and max_image_dim != -1:
        hw_now = raydepth.shape[:2]
        ratio = max_image_dim / max(hw_now[0], hw_now[1])
        if ratio < 1.0:
            h_new = int(round(hw_now[0] * ratio))
            w_new = int(round(hw_now[1] * ratio))
            raydepth = cv2.resize(raydepth, (w_new, h_new))
    return raydepth
#check function
def raydepth2depth(raydepth, K, img_hw):
    K_inv = np.linalg.inv(K)
    h, w = raydepth.shape[0], raydepth.shape[1]
    grids = np.meshgrid(np.arange(w), np.arange(h))
    coords_homo = [grids[0].reshape(-1), grids[1].reshape(-1), np.ones((h*w))]
    coords_homo = np.stack(coords_homo)
    coeffs = np.linalg.norm(K_inv @ coords_homo, axis=0)
    coeffs = coeffs.reshape(h, w)
    depth = raydepth / coeffs
    return depth

#cusom depthloader as  class limap.base.depth_reader_base.BaseDepthReader(filename)
#Each depth loader consists of the file name of the depth image and its width, height and other information, along with the method for loading.
class flowDepthReader(_base.BaseDepthReader):
    def __init__(self, filename, K, img_hw):
        super(flowDepthReader, self).__init__(filename)
        self.K = K
        self.img_hw = img_hw

    def read(self, filename):
        raydepth = read_raydepth(filename, resize_hw=self.img_hw)
        depth = raydepth2depth(raydepth, self.K, self.img_hw)
        return depth

# def read_scene_hypersim(cfg, dataset, scene_id, cam_id=0, load_depth=False):
#     # set scene id
#     dataset.set_scene_id(scene_id)
#     dataset.set_max_dim(cfg["max_image_dim"])

#     # generate image indexes
#     index_list = np.arange(0, cfg["input_n_views"], cfg["input_stride"]).tolist()
#     index_list = dataset.filter_index_list(index_list, cam_id=cam_id)

#     # get image collections
#     K = dataset.K.astype(np.float32)
#     img_hw = [dataset.h, dataset.w]
#     Ts, Rs = dataset.load_cameras(cam_id=cam_id)
#     cameras, camimages = {}, {}
#     cameras[0] = _base.Camera("SIMPLE_PINHOLE", K, cam_id=0, hw=img_hw)
#     for image_id in index_list:
#         pose = _base.CameraPose(Rs[image_id], Ts[image_id])
#         imname = dataset.load_imname(image_id, cam_id=cam_id)
#         camimage = _base.CameraImage(0, pose, image_name=imname)
#         camimages[image_id] = camimage
#     imagecols = _base.ImageCollection(cameras, camimages)

#     if load_depth:
#         # get depths
#         depths = {}
#         for image_id in index_list:
#             depth_fname = dataset.load_raydepth_fname(image_id, cam_id=cam_id)
#             depth = flowDepthReader(depth_fname, K, img_hw)
#             depths[image_id] = depth
#         return imagecols, depths
#     else:
#         return imagecols