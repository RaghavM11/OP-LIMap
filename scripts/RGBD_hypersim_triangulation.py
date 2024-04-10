import os
import sys
import numpy as np
import argparse
from Hypersim import Hypersim
from loader import read_scene_hypersim
import limap.util.config as cfgutils
import limap.runners
import limapio
import limapvis

def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0):
    imagecols = read_scene_hypersim(cfg, dataset, scene_id, cam_id=cam_id, load_depth=False)
    linetracks = limap.runners.line_triangulation(cfg, imagecols)

    # Getting SFM meta infos from Colmap (integration code to be updated)
    sfminfos_colmap_folder, neighbors, ranges = runners.compute_sfminfos(cfg, imagecols)
    
    # Link to triangulation with depth

    # Visualization using Open3D
    # Save tracks
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "alltracks.txt"), linetracks, n_visible_views=4)
    limapio.save_folder_linetracks_with_info(os.path.join(cfg["dir_save"], cfg["output_folder"]), linetracks, config=cfg, imagecols=imagecols)
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(os.path.join(cfg["dir_save"], 'triangulated_lines_nv{0}.obj'.format(cfg["n_visible_views"])), VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))

    # Visualize
    if cfg["visualize"]:
        validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
        for track_id in range(len(validtracks)):
            limapvis.visualize_line_track(imagecols, validtracks[track_id], prefix="track.{0}".format(track_id))
        VisTrack.vis_reconstruction(imagecols, n_visible_views=cfg["n_visible_views"], width=2)

    return linetracks

def parse_config():
    arg_parser = argparse.ArgumentParser(description='triangulate 3d lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/office.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/office.yaml', help='default config file')
    arg_parser.add_argument('--npyfolder', type=str, default=None, help='folder to load precomputed results')
    arg_parser.add_argument('--visualize', action='store_true', help='enable visualization')

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = {'-nv': '--n_visible_views', '-nn': '--n_neighbors', '-sid': '--scene_id'}
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = args.npyfolder or os.path.join("precomputed", "office", cfg["scene_id"])
    cfg["visualize"] = args.visualize
    return cfg

def main():
    cfg = parse_config()
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])

if __name__ == '__main__':
    main()
