import os, sys
import numpy as np

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.append(REPO_DIR.as_posix())
from limap_extension.utils.path_fixer import allow_limap_imports

allow_limap_imports()

from scripts.run_mapping import (get_argparser, parse_config, parse_args, cfg_to_image_collection)

# We should remove this as a default argument as this path will depend on what experiment we're
# evaluating. This should be the same as the config file used to actually run the experiment.
# DEFAULT_EVAL_CONFIG_PATH = REPO_DIR / "cfgs" / "extension_experiments" / "extension_testing_pruning.yml"

import limap.base as _base
import limap.util.io as limapio
import limap_extension.visualize as limapvis

#add rerun and it's dependencies
import limap.structures as _structures

import rerun as rr

import time


def parse_to_config():
    # import argparse
    arg_parser = get_argparser()
    # arg_parser = argparse.ArgumentParser(description='visualize 3d lines')
    # arg_parser.add_argument('-i', '--input_dir', type=str, required=True, help='input line file. Format supported now: .obj, .npy, linetrack folder.')
    arg_parser.add_argument('-nv',
                            '--n_visible_views',
                            type=int,
                            default=2,
                            help='number of visible views')
    arg_parser.add_argument('--visclouds',
                            action="store_true",
                            help="whether to visualize the point clouds")
    #arg_parser.add_argument('--bpt3d_pl', type=str, default=None, help=".npz file for point-line associations") #add for rerun point-line association
    #arg_parser.add_argument('--bpt3d_vp', type=str, default=None, help=".npz file for line-vanishing point associations") #add for rerun
    #arg_parser.add_argument('--segments2d', type=str, default=None, help="directory containing detected 2D segments")# add for rerun
    # arg_parser.add_argument('--imagecols', type=str, default=None, help=".npy file for imagecols")
    #arg_parser.add_argument("--metainfos", type=str, default=None, help=".txt file for neighbors and ranges")
    #arg_parser.add_argument('--mode', type=str, default="rerun", help="[pyvista, open3d, rerun]")  #rerun added
    # arg_parser.add_argument('--use_robust_ranges', default=False, action='store_true', help="whether to use computed robust ranges")
    # arg_parser.add_argument('--scale', type=float, default=1.0, help="scaling both the lines and the camera geometry")
    # arg_parser.add_argument('--cam_scale', type=float, default=1.0, help="scale of the camera geometry")
    # arg_parser.add_argument('--output_dir', type=str, default=None, help="if set, save the scaled lines in obj format")
    # args = arg_parser.parse_args()
    # arg_parser.add_argument('-c',
    #                         '--config_file',
    #                         type=str,
    #                         default=DEFAULT_EVAL_CONFIG_PATH.as_posix(),
    #                         help='config file')
    # arg_parser.add_argument("--default_config_file",type=str,
    #                         default=DEFAULT_BASE_CONFIG_PATH.as_posix(),
    #                         help='default config file')
    # args, unknown = arg_parser.parse_known_args()

    # cfg = cfgutils.load_config(args.config_file, default_path=args.config_file)

    # shortcuts = dict()
    # shortcuts['-nv'] = '--n_visible_views'
    # shortcuts['-nn'] = '--n_neighbors'
    # shortcuts['-sid'] = '--scene_id'
    # cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    # cfg["folder_to_load"] = args.npyfolder
    # if cfg["folder_to_load"] is None:
    #     # cfg["folder_to_load"] = os.path.join("precomputed", "hypersim", cfg["scene_id"])
    #     folder_to_load_base = REPO_DIR / "limap_extension"
    #     folder_to_load_base.mkdir(parents=True, exist_ok=True)
    #     experiment_source = Path(cfg["extension_dataset"]["dataset_path"]).relative_to(constants.DATASET_DIR)
    #     folder_to_load = folder_to_load_base / experiment_source
    #     cfg["folder_to_load"] = folder_to_load.as_posix()

    args, unknown = parse_args(arg_parser)
    cfg = parse_config(args, unknown)

    # cfg["metainfos"] = os.path.join(cfg["output_dir"], cfg["experiment_name"], "metainfos.txt")
    cfg["metainfos"] = None
    cfg["imagecols"] = None
    cfg["use_robust_ranges"] = False
    cfg["mode"] = "rerun"
    cfg["scale"] = 1.0
    cfg["cam_scale"] = 1.0
    # cfg["visclouds"] = args.visclouds
    cfg["visclouds"] = True
    return cfg


def vis_3d_lines(lines, mode="open3d", ranges=None, scale=1.0):
    print("\n\nHERE\n\n")
    if mode == "pyvista":
        limapvis.pyvista_vis_3d_lines(lines, ranges=ranges, scale=scale)
    elif mode == "open3d":
        limapvis.open3d_vis_3d_lines(lines, ranges=ranges, scale=scale)
    elif mode == "rerun":
        limapvis.rerun_vis_3d_lines(lines, ranges=ranges, scale=scale)
    else:
        raise NotImplementedError


def vis_reconstruction(linetracks,
                       imagecols,
                       mode="open3d",
                       n_visible_views=4,
                       ranges=None,
                       scale=1.0,
                       cam_scale=1.0):
    if mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise ValueError(
            "Error! Visualization with cameras is only supported with open3d and rerun.")
    VisTrack.report()
    VisTrack.vis_reconstruction(imagecols,
                                n_visible_views=n_visible_views,
                                ranges=ranges,
                                scale=scale,
                                cam_scale=cam_scale)


def main(cfg):
    lines_output_file = os.path.join(cfg["output_dir"], cfg["experiment_name"], "finaltracks")
    print(lines_output_file)
    lines, linetracks = limapio.read_lines_from_input(lines_output_file)
    print("Lines:", lines)
    print("Linetracks:", linetracks)
    ranges = None
    if cfg["metainfos"] is not None:
        _, ranges = limapio.read_txt_metainfos(cfg["metainfos"])
    if cfg["use_robust_ranges"]:
        ranges = limapvis.compute_robust_range_lines(lines)
    if cfg["n_visible_views"] > 2 and linetracks is None:
        raise ValueError("Error! Track information is not available.")
    if cfg["imagecols"] is None:
        vis_3d_lines(lines, mode=cfg["mode"], ranges=ranges, scale=cfg["scale"])

        if cfg["visclouds"]:
            from limap_extension.point_cloud import PointCloud
            from limap_extension.point_cloud_list import PointCloudList
            from limap_extension.transforms_spatial import get_transform_matrix_from_pose_array
            from limap_extension.visualization.rerun.figure_factory import FigureFactory
            from limap_extension.constants import CAM_INTRINSIC
            from limap_extension.constants import ImageDirection
            from limap_extension.utils.io import read_all_rgbd, read_pose
            from limap_extension.img_cloud_transforms import (reproject_img, uvz_ocv_to_xyz_ned,
                                                              xyz_ned_to_uvz_ocv,
                                                              index_img_with_uv_coords,
                                                              get_uv_coords, imgs_to_clouds_np,
                                                              ned2cam_single_pose, transform_cloud,
                                                              inverse_pose)
            from limap_extension.constants import H_NED_TO_OCV, H_OCV_TO_NED
            trial_dir = cfg["extension_dataset"]["dataset_path"]
            poses_world_frame = read_pose(trial_dir, ImageDirection.LEFT)

            imgs, _, depths, _ = read_all_rgbd(trial_dir, ImageDirection.LEFT)
            pcl = PointCloudList()
            pc: PointCloud = None
            for i, (img, depth) in enumerate(zip(imgs, depths)):
                # Should truly be in camera frame (NED)
                # It is. X corresponds with the depth channel.
                cloud, _ = imgs_to_clouds_np(img, depth, CAM_INTRINSIC)
                cloud.normalize_rgb()

                # May need to do inverse like original transform code.
                # pose_cam_to_world = inverse_pose(ned2cam_single_pose(poses_world_frame[i]))
                # pose_cam_to_world = get_transform_matrix_from_pose_array(poses_world_frame[i])
                # pose_cam_to_world = H_OCV_TO_NED @ inverse_pose(
                #     get_transform_matrix_from_pose_array(poses_world_frame[i])) @ H_NED_TO_OCV
                # Looks good but is off from the lines almost like a reflection
                # pose_cam_to_world = H_NED_TO_OCV @ inverse_pose(
                #     get_transform_matrix_from_pose_array(poses_world_frame[i]))
                # pose_cam_to_world = H_OCV_TO_NED @ inverse_pose(
                #     get_transform_matrix_from_pose_array(poses_world_frame[i]))

                # pose_cam_to_world = H_NED_TO_OCV @ inverse_pose(
                #     get_transform_matrix_from_pose_array(poses_world_frame[i]))

                # pose_cam_to_world = H_NED_TO_OCV @ inverse_pose(
                #     get_transform_matrix_from_pose_array(poses_world_frame[i])) @ H_OCV_TO_NED

                # pose_cam_to_world = inverse_pose(
                #     H_NED_TO_OCV @ get_transform_matrix_from_pose_array(
                #         poses_world_frame[i]) @ H_OCV_TO_NED) @ H_NED_TO_OCV

                # To get
                # pose_wned_to_cam_ocv = get_transform_matrix_from_pose_array(poses_world_frame[i])
                # pose_cam_ocv_to_wned = inverse_pose(pose_wned_to_cam_ocv)
                # pose_cam_ocv_to_wned = H_OCV_TO_NED @ pose_cam_ocv_to_wned @ H_NED_TO_OCV
                # pose_cam_to_world = get_transform_matrix_from_pose_array(poses_world_frame[i])
                # pose_cam_to_world = H_NED_TO_OCV @ pose_cam_to_world @ H_OCV_TO_NED
                # pose_cam_to_world = pose_cam_ocv_to_wned @ pose_cam_to_world

                cloud = transform_cloud(cloud, H_NED_TO_OCV)

                pose_cam_in_world_frame = H_NED_TO_OCV @ get_transform_matrix_from_pose_array(
                    poses_world_frame[i])
                pose_cam_to_world = inverse_pose(pose_cam_in_world_frame)

                # pose_cam_to_world = H_NED_TO_OCV @ inverse_pose(
                #     get_transform_matrix_from_pose_array(poses_world_frame[i]))

                cloud = transform_cloud(cloud, pose_cam_to_world)
                # pcl.append(cloud)
                if pc is None:
                    pc = cloud
                else:
                    # Extend the point cloud
                    pc.xyz = np.concatenate((pc.xyz, cloud.xyz), axis=0)
                    print(pc.xyz.shape)
                    pc.rgb = np.concatenate((pc.rgb, cloud.rgb), axis=0)

                pc.downsample(0.10)

            # pc = transform_cloud(pc, H_OCV_TO_NED)
            # pc.xyz = tform_coords
            # pc.xyz = np.dot(H_OCV_TO_NED[:3, :3], pc.xyz.T).T + H_OCV_TO_NED[:3, 3]

            pcl.append(pc)
            ff = FigureFactory()
            ff.add_cloud("Clouds", pcl)
            print(f"Scene bounds max: {ff._scene_bounds_max}")
            print(f"Scene bounds min: {ff._scene_bounds_min}")
            bounds = np.stack((ff._scene_bounds_min, ff._scene_bounds_max), axis=1)
            # mean = np.mean(bounds, axis=1)
            # std = np.std(bounds, axis=1)
            # ff._update_bounds(mean - std, mean + std)
            ff._update_bounds(np.array((-25, -25, -25)), np.array((25, 25, 25)))
            # ff._scene_bounds_min = mean - 2 * std
            # ff._scene_bounds_max = mean + 2 * std
            ff.make_fig(do_init=False)
    else:
        # if (not cfg["imagecols"]):
        #     raise ValueError("Error! Input file {0} is not valid".format(cfg["imagecols"]))
        # imagecols = _base.ImageCollection(limapio.read_npy(cfg["imagecols"]).item())
        imagecols = cfg["imagecols"]
        vis_reconstruction(linetracks,
                           imagecols,
                           mode=cfg["mode"],
                           n_visible_views=cfg["n_visible_views"],
                           ranges=ranges,
                           scale=cfg["scale"],
                           cam_scale=cfg["cam_scale"],
                           segments2d=segments2d)

        if cfg["segments2d"] is None:
            segments2d = None
        else:
            segments2d = limapio.read_all_segments_from_folder(lines_output_file)
    if cfg["dir_save"] is not None:
        out_dir_experiment = os.path.join(cfg["dir_save"], cfg["experiment_name"], "out.obj")
        limapio.save_obj(out_dir_experiment, lines)


if __name__ == '__main__':
    cfg = parse_to_config()
    main(cfg)
