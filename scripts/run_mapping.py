"""Runs the line mapping portion of LIMAP with the ability to prune dynamic objects from scenes via
optical flow.

File based on limap/runners/hypersim/triangulation.py"""

import sys
import os
from pathlib import Path

# Handle all of the path manipulation to find LIMAP modules.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.append(SCRIPTS_DIR.as_posix())
from path_fixer import REPO_DIR, allow_limap_imports

allow_limap_imports()

import limap
import limap.util.config as cfgutils
from Hypersim import Hypersim
from loader import read_scene_hypersim

# from limap_extension.line_triangulation import line_triangulation

# TODO: Create a directory for storing all of our experiment configuration files so that the
# experiments are totally reproducible.

# This is the config file that Shlok and Dylan were working on. Based on the original limap
# triangulation config file but with added info for tartainair/optical flow.
DEFAULT_CONFIG_PATH = REPO_DIR / "cfgs" / "default.yaml"

# I believe this is the config file that defines the base configuration. Any values specified in the
# "--config-file" argument will override the values in this configuration when running LIMAP.
DEFAULT_BASE_CONFIG_PATH = REPO_DIR / "cfgs" / "triangulation" / "default.yaml"


def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0):
    imagecols = read_scene_hypersim(cfg, dataset, scene_id, cam_id=cam_id, load_depth=False)
    linetracks = limap.runners.line_triangulation(cfg, imagecols)
    return linetracks


def parse_config():
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

    args, unknown = arg_parser.parse_known_args()

    print("parsing")
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    print("Done parsing")
    print("cfg:", cfg)

    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    shortcuts['-sid'] = '--scene_id'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = args.npyfolder
    if cfg["folder_to_load"] is None:
        cfg["folder_to_load"] = os.path.join("precomputed", "hypersim", cfg["scene_id"])
    return cfg


def main():
    cfg = parse_config()
    # print(cfg)
    # return

    # TODO: It's up to group members to decide if we need to/want to run HyperSim or instead fake
    # the run with our ground truth information.
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])


if __name__ == '__main__':
    main()