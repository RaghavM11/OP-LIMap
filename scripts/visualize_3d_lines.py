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
import limap.visualize as limapvis

def parse_to_config():
    # import argparse
    arg_parser = get_argparser()
    # arg_parser = argparse.ArgumentParser(description='visualize 3d lines')
    # arg_parser.add_argument('-i', '--input_dir', type=str, required=True, help='input line file. Format supported now: .obj, .npy, linetrack folder.')
    arg_parser.add_argument('-nv', '--n_visible_views', type=int, default=2, help='number of visible views')
    # arg_parser.add_argument('--imagecols', type=str, default=None, help=".npy file for imagecols")
    #arg_parser.add_argument("--metainfos", type=str, default=None, help=".txt file for neighbors and ranges")
    # arg_parser.add_argument('--mode', type=str, default="open3d", help="[pyvista, open3d]")
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
        
    cfg["metainfos"] = os.path.join(cfg["output_dir"], cfg["experiment_name"], "metainfos.txt")
    cfg["imagecols"] = cfg_to_image_collection(cfg)
    cfg["use_robust_ranges"] = False
    cfg["mode"] = "open3d"
    cfg["scale"] = 1.0
    cfg["cam_scale"] = 1.0
    return cfg

def vis_3d_lines(lines, mode="open3d", ranges=None, scale=1.0):
    if mode == "pyvista":
        limapvis.pyvista_vis_3d_lines(lines, ranges=ranges, scale=scale)
    elif mode == "open3d":
        limapvis.open3d_vis_3d_lines(lines, ranges=ranges, scale=scale)
    else:
        raise NotImplementedError

def vis_reconstruction(linetracks, imagecols, mode="open3d", n_visible_views=4, ranges=None, scale=1.0, cam_scale=1.0):
    if mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise ValueError("Error! Visualization with cameras is only supported with open3d.")
    VisTrack.report()
    VisTrack.vis_reconstruction(imagecols, n_visible_views=n_visible_views, ranges=ranges, scale=scale, cam_scale=cam_scale)

# Npy_linematching = REPO_DIR / "outputs" / "triangulation"/"line_matchings"

def main(cfg):
    lines_output_file = os.path.join(cfg["output_dir"], cfg["experiment_name"], "finaltracks")#, "all_2d_segs.npy")
    lines, linetracks = limapio.read_lines_from_input(lines_output_file)
    ranges = None
    if cfg["metainfos"] is not None:
        _, ranges = limapio.read_txt_metainfos(cfg["metainfos"])
    if cfg["use_robust_ranges"]:
        ranges = limapvis.compute_robust_range_lines(lines)
    if cfg["n_visible_views"] > 2 and linetracks is None:
        raise ValueError("Error! Track information is not available.")
    if cfg["imagecols"] is None:
        vis_3d_lines(lines, mode=cfg["mode"], ranges=ranges, scale=cfg["scale"])
    else:
        # if (not cfg["imagecols"]):
        #     raise ValueError("Error! Input file {0} is not valid".format(cfg["imagecols"]))
        # imagecols = _base.ImageCollection(limapio.read_npy(cfg["imagecols"]).item())
        imagecols = cfg["imagecols"]
        vis_reconstruction(linetracks, imagecols, mode=cfg["mode"], n_visible_views=cfg["n_visible_views"], ranges=ranges, scale=cfg["scale"], cam_scale=cfg["cam_scale"])
    if cfg["dir_save"] is not None:
        out_dir_experiment = os.path.join(cfg["dir_save"], cfg["experiment_name"])
        limapio.save_obj(out_dir_experiment, lines)

if __name__ == '__main__':
    cfg = parse_to_config()
    main(cfg)

