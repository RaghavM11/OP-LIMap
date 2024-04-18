import os
import numpy as np
from tqdm import tqdm

import limap.base as _base
import limap.util.io as limapio
import limap.visualize as limapvis

def run_saved_tracks(cfg, linetracks, imagecols, all_2d_segs):
    # Save tracks
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "alltracks.txt"), linetracks, n_visible_views=4)
    limapio.save_folder_linetracks_with_info(os.path.join(cfg["dir_save"], cfg["output_folder"]), linetracks, config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(os.path.join(cfg["dir_save"], 'triangulated_lines_nv{0}.obj'.format(cfg["n_visible_views"])), VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))
    
    # Visualize
    if cfg["visualize"]:
        validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
        def report_track(track_id):
            limapvis.visualize_line_track(imagecols, validtracks[track_id], prefix="track.{0}".format(track_id))
        VisTrack.vis_reconstruction(imagecols, n_visible_views=cfg["n_visible_views"], width=2)

    return linetracks

# Example usage:
# Define your cfg, linetracks, imagecols, and all_2d_segs here
# cfg = {...}
# linetracks = [...]
# imagecols = [...]
# all_2d_segs = [...]

# Call the function to run the saved tracks
#run_saved_tracks(cfg, linetracks, imagecols, all_2d_segs)

