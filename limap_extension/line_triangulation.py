from typing import Dict
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2 
import limap.base as _base
import limap.merging as _mrg
import limap.triangulation as _tri
import limap.vplib as _vplib
import limap.pointsfm as _psfm
import limap.optimize as _optim
import limap.runners as _runners
import limap.util.io as limapio
import limap.visualize as limapvis
import PIL
from PIL import GifImagePlugin
from PIL import Image

# from limap_extension.

#reading mask

#read all 100 mask using PIL and convert to numpy array
def mask_to_array():
    mask_arrays = []
    directory_path="/home/mr/Desktop/Navarch 568/Project/LIMap-Extension/datasets/carwelding_Hard_P001_with_flow_masks-002/carwelding/Hard/P001/dynamic_obj_masks_left/flow_xyz_method"
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):  # Assuming the masks are PNG files
            # Construct the full path to the image file
            full_path = os.path.join(directory_path, filename)
            # Open the image file using PIL
            img = Image.open(full_path)
            # Convert the image to a numpy array
            mask_array = np.array(img)
            # Append the mask array to the list
            mask_arrays.append(mask_array)
    return np.array(mask_arrays)


def read_calc_fake_sfm_data(cfg: Dict):
    root_path = Path(cfg["extension_dataset"]["dataset_path"])

    if cfg["extension_dataset"]["cam_id"] != "left":
        raise NotImplementedError("Only left camera is supported for now")

    # Read the pose information
    # TODO: Decide on pose file name based on cam id
    pose_path = root_path / "pose_left.txt"
    poses = np.loadtxt(pose_path)

    # Get the sequence of images.
    img_dir = root_path / "image_left"
    img_idxs = []
    for img_path in img_dir.iterdir():
        img_idx = int(img_path.stem.split("_")[1])
        img_idxs.append(img_idx)
    img_idxs.sort()

    n_neighbors = cfg["n_neighbors"]
    neighbors = dict()
    for img_idx in img_idxs:
        neighbors[img_idx] = []
        for i in range(1, n_neighbors + 1):
            if img_idx - i in img_idxs:
                neighbors[img_idx].append(img_idx - i)
            if img_idx + i in img_idxs:
                neighbors[img_idx].append(img_idx + i)

    # TODO: Load info from config file and calculate info we'd get from SfM (neighbors, ranges).
    # Neighbors is a dictionary with image_id as key and a list of neighbor image_ids as value.
    # - We should be able to get this just from the sequence of images, e.g. if K is 3, then
    #   image 1 has neighbors [2, 3], image 2 has neighbors [1, 3], and image 3 has neighbors [1,
    #   2].
    neighbors = dict()

    # Ranges is a tuple of numpy arrays indicating the bounding box of the scene.
    # TODO: Calculate this from the poses and a buffer.
    # - If we want to get fancy, we could try to use the depth information to do this for us but I'd
    #   rather not do that.
    ranges = (np.array([0, 0, 0]), np.array([1, 1, 1]))

    return neighbors, ranges


def line_triangulation(cfg, imagecols, neighbors=None, ranges=None):
    '''
    Main interface of line triangulation over multi-view images.

    Args:
        cfg (dict): Configuration. Fields refer to :file:`cfgs/triangulation/default.yaml` as an example
        imagecols (:class:`limap.base.ImageCollection`): The image collection corresponding to all the images of interest
        neighbors (dict[int -> list[int]], optional): visual neighbors for each image. By default we compute neighbor information from the covisibility of COLMAP triangulation.
        ranges (pair of :class:`np.array` each of shape (3,), optional): robust 3D ranges for the scene. By default we compute range information from the COLMAP triangulation.
    Returns:
        list[:class:`limap.base.LineTrack`]: list of output 3D line tracks
    '''
    print("[LOG] Number of images: {0}".format(imagecols.NumImages()))
    cfg = _runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["triangulation"]["var2d"] == -1:
        cfg["triangulation"]["var2d"] = cfg["var2d"][detector_name]
    # undistort images
    if not imagecols.IsUndistorted():
        imagecols = _runners.undistort_images(imagecols,
                                              os.path.join(cfg["dir_save"],
                                                           cfg["undistortion_output_dir"]),
                                              skip_exists=cfg["load_undistort"]
                                              or cfg["skip_exists"],
                                              n_jobs=cfg["n_jobs"])
    # resize cameras
    assert imagecols.IsUndistorted() == True
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])
    limapio.save_txt_imname_dict(os.path.join(cfg["dir_save"], 'image_list.txt'),
                                 imagecols.get_image_name_dict())
    limapio.save_npy(os.path.join(cfg["dir_save"], 'imagecols.npy'), imagecols.as_dict())

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    # sfminfos_colmap_folder = None
    # if neighbors is None:
    #     sfminfos_colmap_folder, neighbors, ranges = _runners.compute_sfminfos(cfg, imagecols)
    # else:
    #     limapio.save_txt_metainfos(os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors,
    #                                ranges)
    #     neighbors = imagecols.update_neighbors(neighbors)
    #     for img_id, neighbor in neighbors.items():
    #         neighbors[img_id] = neighbors[img_id][:cfg["n_neighbors"]]
    # limapio.save_txt_metainfos(os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges)

    if neighbors is not None:
        raise ValueError("neighbors shouldn't be specified. They're calculated in this funciton")
    if ranges is not None:
        raise ValueError("ranges shouldn't be specified. They're calculated in this funciton")

    neighbors, ranges = read_calc_fake_sfm_data()

    ##########################################################
    # [B] get 2D line segments for each image and prune them
    ##########################################################
    compute_descinfo = (not cfg["triangulation"]["use_exhaustive_matcher"])
    compute_descinfo = (compute_descinfo and (not cfg["load_match"]) and
                        (not cfg["load_det"])) or cfg["line2d"]["compute_descinfo"]
    all_2d_segs, descinfo_folder = _runners.compute_2d_segs(cfg,
                                                            imagecols,
                                                            compute_descinfo=compute_descinfo)

    print("\n\nINSERT LINE PRUNING HERE\n\n")
    # read the masks in from the config file
    # assuming that all the masks from frames 1 - N are stored in masks as a NxHxWx1 matrix
    #masks = cfg["masks"]
    masks= mask_to_array()
    # N is the number of frames
    N = masks.shape[0]  # dummy need to change
    for i in range(1, N):
        mask = masks[i]
        # fing dynamic object pixels in the mask and remove them from the 2d segments
        # dynamic object pixels are denotes as 1's in the mask
        segment = all_2d_segs[i]
        # find the dynamic object pixels
        dynamic_object_pixels = np.where(mask == 1)
        for pixel in dynamic_object_pixels:
            x, y = pixel
            x1 = segment[:, 0]
            y1 = segment[:, 1]
            x2 = segment[:, 2]
            y2 = segment[:, 3]

            match_x1 = np.where(x1 == x)
            match_x2 = np.where(x2 == x)
            match_y1 = np.where(y1 == y)
            match_y2 = np.where(y2 == y)

            match1 = np.intersect1d(match_x1, match_y1)
            match2 = np.intersect1d(match_x2, match_y2)

            matching_segments = np.union1d(match1, match2)
            # remove the matching segments
            for idx in matching_segments:
                segment = np.delete(segment, idx, axis=0)
        
        all_2d_segs[i] = segment
        # remove the dynamic object pixels from the 2d segments
        # for pixel in dynamic_object_pixels:
        #     x, y = pixel
        #     # Note: the condition below may need to be changed because I do not know if the in
        #     # operation works on arays or not
        #     # .     In case it throws an error  just check individually
        #     # Note: segment is a Nx4/5 array where N is the number of segments and each segment is a
        #     # 4/5 tuple of x1, y1, x2, y2, [score]
        #     # check if the location is in the segments already if so, remove the entire segment
        #     x1 = segment[:, 0]
        #     y1 = segment[:, 1]
        #     x2 = segment[:, 2]
        #     y2 = segment[:, 3]
        #     # check if the pixel is in the segment
        #     # if it is, remove the segments 
        #     # remove the matching segments
        #     for idx in matching_segments:
        #         segment = np.delete(segment, idx, axis=0)
            
        
    # All 2d segs is likely an iterable where each item somehow indicates a line in the image
    # (likely (pt_1, pt_2)).
    # idxs_to_keep = []
    # for i, line in enumerate(all_2d_segs):
    #     # Check if the line intersects or overlaps with any area of the mask that is associated with
    #     # dynamic objects.
    #     is_line_associated_with_dynamic_object = do_your_method(line, masks)

    #     if not is_line_associated_with_dynamic_object:
    #         idxs_to_keep.append(i)

    # Delete the lines that are associated with dynamic objects
    # for idx in idxs_to_keep:

    ##########################################################
    # [C] get line matches
    ##########################################################
    if not cfg["triangulation"]["use_exhaustive_matcher"]:
        matches_dir = _runners.compute_matches(cfg, descinfo_folder, imagecols.get_img_ids(),
                                               neighbors)

    ##########################################################
    # [D] multi-view triangulation
    ##########################################################
    Triangulator = _tri.GlobalLineTriangulator(cfg["triangulation"])
    Triangulator.SetRanges(ranges)
    all_2d_lines = _base.get_all_lines_2d(all_2d_segs)
    Triangulator.Init(all_2d_lines, imagecols)
    if cfg["triangulation"]["use_vp"]:
        vpdetector = _vplib.get_vp_detector(cfg["triangulation"]["vpdet_config"],
                                            n_jobs=cfg["triangulation"]["vpdet_config"]["n_jobs"])
        vpresults = vpdetector.detect_vp_all_images(all_2d_lines, imagecols.get_map_camviews())
        Triangulator.InitVPResults(vpresults)

    # get 2d bipartites from pointsfm model
    # if cfg["triangulation"]["use_pointsfm"]["enable"]:
    #     if cfg["triangulation"]["use_pointsfm"]["colmap_folder"] is None:
    #         colmap_model_path = None
    #         # check if colmap model exists from sfminfos computation
    #         if cfg["triangulation"]["use_pointsfm"][
    #                 "reuse_sfminfos_colmap"] and sfminfos_colmap_folder is not None:
    #             colmap_model_path = os.path.join(sfminfos_colmap_folder, "sparse")
    #             if not _psfm.check_exists_colmap_model(colmap_model_path):
    #                 colmap_model_path = None
    #         # retriangulate
    #         if colmap_model_path is None:
    #             colmap_output_path = os.path.join(cfg["dir_save"], "colmap_outputs_junctions")
    #             input_neighbors = None
    #             if cfg["triangulation"]["use_pointsfm"]["use_neighbors"]:
    #                 input_neighbors = neighbors
    #             _psfm.run_colmap_sfm_with_known_poses(cfg["sfm"],
    #                                                   imagecols,
    #                                                   output_path=colmap_output_path,
    #                                                   skip_exists=cfg["skip_exists"],
    #                                                   neighbors=input_neighbors)
    #             colmap_model_path = os.path.join(colmap_output_path, "sparse")
    #     else:
    #         colmap_model_path = cfg["triangulation"]["use_pointsfm"]["colmap_folder"]
    #     reconstruction = _psfm.PyReadCOLMAP(colmap_model_path)
    #     all_bpt2ds, sfm_points = _runners.compute_2d_bipartites_from_colmap(
    #         reconstruction, imagecols, all_2d_lines, cfg["structures"]["bpt2d"])
    #     Triangulator.SetBipartites2d(all_bpt2ds)
    #     if cfg["triangulation"]["use_pointsfm"]["use_triangulated_points"]:
    #         Triangulator.SetSfMPoints(sfm_points)

    # triangulate
    print('Start multi-view triangulation...')
    for img_id in tqdm(imagecols.get_img_ids()):
        if cfg["triangulation"]["use_exhaustive_matcher"]:
            Triangulator.TriangulateImageExhaustiveMatch(img_id, neighbors[img_id])
        else:
            matches = limapio.read_npy(os.path.join(matches_dir,
                                                    "matches_{0}.npy".format(img_id))).item()
            Triangulator.TriangulateImage(img_id, matches)
    linetracks = Triangulator.ComputeLineTracks()

    # filtering 2d supports
    linetracks = _mrg.filtertracksbyreprojection(
        linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"],
        cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    if not cfg["triangulation"]["remerging"]["disable"]:
        # remerging
        linker3d = _base.LineLinker3d(cfg["triangulation"]["remerging"]["linker3d"])
        linetracks = _mrg.remerge(linker3d, linetracks)
        linetracks = _mrg.filtertracksbyreprojection(
            linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"],
            cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    linetracks = _mrg.filtertracksbysensitivity(
        linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_sv_angular_3d"],
        cfg["triangulation"]["filtering2d"]["th_sv_num_supports"])
    linetracks = _mrg.filtertracksbyoverlap(
        linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_overlap"],
        cfg["triangulation"]["filtering2d"]["th_overlap_num_supports"])
    validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]

    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]:
        cfg_ba = _optim.HybridBAConfig(cfg["refinement"])
        cfg_ba.set_constant_camera()
        ba_engine = _optim.solve_line_bundle_adjustment(cfg["refinement"],
                                                        imagecols,
                                                        linetracks,
                                                        max_num_iterations=200)
        linetracks_map = ba_engine.GetOutputLineTracks(
            num_outliers=cfg["refinement"]["num_outliers_aggregator"])
        linetracks = [track for (track_id, track) in linetracks_map.items()]

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "alltracks.txt"),
                                linetracks,
                                n_visible_views=4)
    limapio.save_folder_linetracks_with_info(os.path.join(cfg["dir_save"], cfg["output_folder"]),
                                             linetracks,
                                             config=cfg,
                                             imagecols=imagecols,
                                             all_2d_segs=all_2d_segs)
    # if is_visualizing:
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(
        os.path.join(cfg["dir_save"],
                     'triangulated_lines_nv{0}.obj'.format(cfg["n_visible_views"])),
        VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))

    # visualize
    if cfg["visualize"]:
        validtracks = [
            track for track in linetracks if track.count_images() >= cfg["n_visible_views"]
        ]

        def report_track(track_id):
            limapvis.visualize_line_track(imagecols,
                                          validtracks[track_id],
                                          prefix="track.{0}".format(track_id))

        import pdb
        pdb.set_trace()
        VisTrack.vis_reconstruction(imagecols, n_visible_views=cfg["n_visible_views"], width=2)
        pdb.set_trace()
    return linetracks
