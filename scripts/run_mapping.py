"""Runs the line mapping portion of LIMAP with the ability to prune dynamic objects from scenes via
optical flow.

File based on limap/runners/hypersim/triangulation.py"""

import sys
import os
from pathlib import Path
from typing import Dict

# Handle all of the path manipulation to find LIMAP modules.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.append(SCRIPTS_DIR.as_posix())
from path_fixer import REPO_DIR, allow_limap_imports

allow_limap_imports()

import numpy as np

import limap
import limap_extension as lext
import limap.base as _base
import limap_extension.utils.config as cfgutils

from limap_extension import constants
from limap_extension.utils.io import read_all_rgbd, read_pose
from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array
from limap_extension.line_triangulation import line_triangulation
# from limap_extension.img_cloud_transforms import cam2ned_single_pose, inverse_pose
from limap_extension.img_cloud_transforms import ned2cam_single_pose
# from limap_extension.line_triangulation import line_triangulation

# TODO: Create a directory for storing all of our experiment configuration files so that the
# experiments are totally reproducible.

# This is the config file that Shlok and Dylan were working on. Based on the original limap
# triangulation config file but with added info for tartainair/optical flow.
# DEFAULT_CONFIG_PATH = REPO_DIR / "cfgs" / "default.yaml"
DEFAULT_CONFIG_PATH = REPO_DIR / "cfgs" / "extension_experiments" / "extension_testing_pruning.yml"
# DEFAULT_CONFIG_PATH = REPO_DIR / "cfgs" / "triangulation" / "extension_testing_no_pruning.yml"

# I believe this is the config file that defines the base configuration. Any values specified in the
# "--config-file" argument will override the values in this configuration when running LIMAP.
DEFAULT_BASE_CONFIG_PATH = REPO_DIR / "cfgs" / "extension_experiments" / "default.yaml"

# import Hypersim

# HYPERSIM_LOADER_PATH = REPO_DIR / "limap" / "runners" / "hypersim"
# sys.path.append(HYPERSIM_LOADER_PATH.as_posix())
# from loader import read_scene_hypersim


def cfg_to_image_collection(cfg: Dict):
    K = constants.CAM_INTRINSIC.astype(np.float32)
    img_hw = [480, 640]

    images, image_name, _, _ = read_all_rgbd(cfg["extension_dataset"]["dataset_path"],
                                             constants.ImageDirection.LEFT)
    max_num_imgs = cfg["extension_dataset"].get("max_num_imgs", None)
    if max_num_imgs is None:
        num_imgs_to_use = len(images)
    else:
        num_imgs_to_use = min(max_num_imgs, len(images))
    # num_imgs_to_read = len(images)

    cam_pose = read_pose(cfg["extension_dataset"]["dataset_path"], constants.ImageDirection.LEFT)
    cam_ext = []
    for pose_cam_in_world_frame in cam_pose[:num_imgs_to_use]:
        # The pose returned from read_pose is in world coordinate frame, which differs
        # from the camera coordinate frame that the intrinsic expects. To fix this, we need to
        # convert the world pose into the camera frame (pose in NED frame).
        pose_cam_in_world_frame = ned2cam_single_pose(pose_cam_in_world_frame)
        # cam_ext.append(get_transform_matrix_from_pose_array(pose_cam_in_world_frame))
        cam_ext.append(pose_cam_in_world_frame)

    cameras, camimages = {}, {}
    cameras[0] = _base.Camera("SIMPLE_PINHOLE", K, cam_id=0, hw=img_hw)

    for image_id in range(num_imgs_to_use):
        pose_cam_in_world_frame = _base.CameraPose(cam_ext[image_id][:3, :3], cam_ext[image_id][:3,
                                                                                                3])
        imname = image_name[image_id]
        camimage = _base.CameraImage(0, pose_cam_in_world_frame, image_name=imname)
        camimages[image_id] = camimage
    imagecols = _base.ImageCollection(cameras, camimages)
    return imagecols


def rub_scene_tartanair_pruning(cfg, cam_id=0):
    imagecols = cfg_to_image_collection(cfg)
    linetracks = line_triangulation(cfg, imagecols)
    return linetracks


def run_scene_tartanair(cfg, cam_id=0):
    K = constants.CAM_INTRINSIC.astype(np.float32)
    img_hw = [480, 640]
    images, image_name, _, _ = read_all_rgbd(cfg["extension_dataset"]["dataset_path"],
                                             constants.ImageDirection.LEFT)
    cam_pose = read_pose(cfg["extension_dataset"]["dataset_path"], constants.ImageDirection.LEFT)
    cam_ext = []
    for pose in cam_pose:
        cam_ext.append(get_transform_matrix_from_pose_array(pose))

    cameras, camimages = {}, {}
    cameras[0] = _base.Camera("SIMPLE_PINHOLE", K, cam_id=0, hw=img_hw)
    for image_id in range(images.shape[0]):
        pose = _base.CameraPose(cam_ext[image_id][:3, :3], cam_ext[image_id][:3, 3])
        imname = image_name[image_id]
        camimage = _base.CameraImage(0, pose, image_name=imname)
        camimages[image_id] = camimage
    imagecols = _base.ImageCollection(cameras, camimages)
    linetracks = limap.runners.line_triangulation(cfg, imagecols)
    return linetracks


def get_argparser():
    import argparse
    arg_parser = argparse.ArgumentParser(description='triangulate 3d lines')

    arg_parser.add_argument('-c',
                            '--config_file',
                            type=str,
                            default=DEFAULT_CONFIG_PATH.as_posix(),
                            help='config file')
    arg_parser.add_argument('--default_config_file',
                            type=str,
                            default=DEFAULT_BASE_CONFIG_PATH.as_posix(),
                            help='default config file')
    arg_parser.add_argument('--npyfolder',
                            type=str,
                            default=None,
                            help='folder to load precomputed results')
    return arg_parser


def parse_args(arg_parser):
    if arg_parser is None:
        arg_parser = get_argparser()

    args, unknown = arg_parser.parse_known_args()
    return args, unknown


def parse_config(args=None, unknown=None, arg_parser=None):
    if args is None and unknown is None:
        if arg_parser is None:
            arg_parser = get_argparser()
        args, unknown = parse_args(arg_parser)

    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)

    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    shortcuts['-sid'] = '--scene_id'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = args.npyfolder
    if cfg["folder_to_load"] is None:
        # cfg["folder_to_load"] = os.path.join("precomputed", "hypersim", cfg["scene_id"])
        folder_to_load_base = REPO_DIR / "precomputed" / "limap_extension"
        folder_to_load_base.mkdir(parents=True, exist_ok=True)
        experiment_source = Path(cfg["extension_dataset"]["dataset_path"]).relative_to(
            constants.DATASET_DIR)
        folder_to_load = folder_to_load_base / experiment_source
        cfg["folder_to_load"] = folder_to_load.as_posix()
    return cfg


def main():
    cfg = parse_config()
    # print(cfg)
    # return
    # TODO: Need to create an easy method of running LI-MAP both with and without the optical flow
    # pruning.
    # dataset = Hypersim(cfg["data_dir"])
    # run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])
    # LIMAP_DIR = REPO_DIR / "limap"
    # LIMAP_DIR.cwd()
    rub_scene_tartanair_pruning(cfg, cam_id=0)


if __name__ == '__main__':
    main()
