from pathlib import Path
from typing import List, Optional, Union
import requests
import os
from tqdm import tqdm
import zipfile
import io
import numpy as np

import cv2

# NOTEBOOK_DIR = Path(".").resolve()
ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parents[1]
# print(REPO_DIR)

# Directories to save/extract downloaded files
DOWNLOAD_DIR = REPO_DIR / "datasets"
EXTRACT_ROOT = REPO_DIR / "datasets"

SCENARIO = "carwelding"
DIFFICULTY = "Hard"


def get_urls(scenario, difficulty):
    '''
    Get the URLs for the specified scenario and difficulty
    '''
    urls = [
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/depth_left.zip",
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/depth_right.zip",
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/flow_flow.zip",
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/flow_mask.zip",
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/image_left.zip",
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/image_right.zip",
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/seg_left.zip",
        f"https://airlab-share.andrew.cmu.edu:8081/tartanair/{scenario}/{difficulty}/seg_right.zip",
    ]
    return urls


def download_zips(urls: List[str]) -> List[Path]:
    # Download each file
    downloaded_zips = []
    for url in urls:
        # Get the filename from the URL
        filename = DOWNLOAD_DIR / url.split("/")[-1]

        if filename.exists():
            print(f"File {filename} already exists. Skipping download.")
            downloaded_zips.append(filename)
            continue

        # Send a GET request to download the file
        response = requests.get(url, stream=True)

        # Get the file size from the headers
        file_size = int(response.headers.get('content-length', 0))

        # Initialize tqdm with the file size
        progress_bar = tqdm(total=file_size,
                            unit='B',
                            unit_scale=True,
                            desc=filename.as_posix(),
                            leave=True)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Write the content of the response to a file
            with filename.open('wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    # Update the progress bar
                    progress_bar.update(len(data))
            # Close the progress bar
            progress_bar.close()
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")

        downloaded_zips.append(filename)
    return downloaded_zips


def _extract_zip_no_filter(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def _extract_zip_with_filter(zip_file, extract_to, trial_filter: Union[List[str], str]):
    if isinstance(trial_filter, str):
        trial_filter = [trial_filter]

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for info in zip_ref.namelist():
            if any([info.startswith(filt) for filt in trial_filter]):
                zip_ref.extract(info, path=extract_to)


def extract_zip(zip_file, extract_to, trial_filter: Optional[Union[List[str], str]] = None):
    if trial_filter is not None:
        _extract_zip_with_filter(zip_file, extract_to, trial_filter)
    else:
        _extract_zip_no_filter(zip_file, extract_to)


# List of URLs to download
urls = get_urls(SCENARIO, DIFFICULTY)

# Create the directories if they doesn't exist
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

downloaded_zips = download_zips(urls)

# yapf: disable
trial_filter = [
    f"{SCENARIO}/{DIFFICULTY}/P005",
    f"{SCENARIO}/{DIFFICULTY}/P006",
    f"{SCENARIO}/{DIFFICULTY}/P007"
]
# yapf: enable

# Extract each downloaded file
for zip_file in downloaded_zips:
    # Extract the downloaded zip file
    # extract_dir = EXTRACT_ROOT / zip_file.stem
    extract_dir = EXTRACT_ROOT
    extract_zip(zip_file, extract_dir, trial_filter)
    print(f"Extracted {zip_file}")

# def read_numpy_file(numpy_file):
#     '''
#     return a numpy array given the file path
#     '''
#     # Open the file in binary mode
#     with open(numpy_file, 'rb') as f:
#         ee = io.BytesIO(f.read())
#         ff = np.load(ee)
#     return ff

# def read_image_file(image_file):
#     '''
#     return a uint8 numpy array given the file path
#     '''
#     # Read the image using OpenCV
#     img = cv2.imread(image_file, cv2.IMREAD_COLOR)
#     if img is None:
#         raise FileNotFoundError(f"Image file '{image_file}' not found or could not be opened.")

#     # Convert BGR to RGB
#     im_rgb = img[:, :, [2, 1, 0]]  # BGR2RGB
#     return im_rgb

# def depth2vis(depth, maxthresh=50):
#     depthvis = np.clip(depth, 0, maxthresh)
#     depthvis = depthvis / maxthresh * 255
#     depthvis = depthvis.astype(np.uint8)
#     depthvis = np.tile(depthvis.reshape(depthvis.shape + (1, )), (1, 1, 3))
#     return depthvis

# def seg2vis(segnp):
#     colors = np.loadtxt('seg_rgbs.txt')
#     segvis = np.zeros(segnp.shape + (3, ), dtype=np.uint8)

#     for k in range(256):
#         mask = segnp == k
#         colorind = k % len(colors)
#         if np.sum(mask) > 0:
#             segvis[mask, :] = colors[colorind]

#     return segvis

# def _calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
#     a = np.arctan2(dv, du)

#     angleShift = np.pi

#     if (True == flagDegree):
#         a = a / np.pi * 180
#         angleShift = 180
#         # print("Convert angle from radian to degree as demanded by the input file.")

#     d = np.sqrt(du * du + dv * dv)

#     return a, d, angleShift

# def flow2vis(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0):
#     """
#     Show an optical flow field as the KITTI dataset does.
#     Some parts of this function are the transform of the original MATLAB code flow_to_color.m.
#     """

#     ang, mag, _ = _calculate_angle_distance_from_du_dv(flownp[:, :, 0],
#                                                        flownp[:, :, 1],
#                                                        flagDegree=False)

#     # Use Hue, Saturation, Value color model
#     hsv = np.zeros((ang.shape[0], ang.shape[1], 3), dtype=np.float32)

#     am = ang < 0
#     ang[am] = ang[am] + np.pi * 2

#     hsv[:, :, 0] = np.remainder((ang + angShift) / (2 * np.pi), 1)
#     hsv[:, :, 1] = mag / maxF * n
#     hsv[:, :, 2] = (n - hsv[:, :, 1]) / n

#     hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 1) * hueMax
#     hsv[:, :, 1:3] = np.clip(hsv[:, :, 1:3], 0, 1) * 255
#     hsv = hsv.astype(np.uint8)

#     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#     if (mask is not None):
#         mask = mask > 0
#         rgb[mask] = np.array([0, 0, 0], dtype=np.uint8)

#     return rgb

# # %%
# import os
# import matplotlib.pyplot as plt
# import time

# # Assuming left_img_folder and right_img_folder are the paths to the folders containing left and right images respectively
# left_img_folder = "/path/to/left/image/folder"
# right_img_folder = "/path/to/right/image/folder"

# left_img_list = sorted(os.listdir(left_img_folder))
# right_img_list = sorted(os.listdir(right_img_folder))

# plt.ion()  # Turn on interactive mode

# for data_ind in range(len(left_img_list)):  # Loop through all images in the folder
#     # Read left and right images using read_image_file function
#     left_img = read_image_file(os.path.join(left_img_folder, left_img_list[data_ind]))
#     right_img = read_image_file(os.path.join(right_img_folder, right_img_list[data_ind]))

#     # Plotting
#     plt.figure(figsize=(12, 5))
#     plt.subplot(121)
#     plt.imshow(left_img)
#     plt.title('Left Image')
#     plt.subplot(122)
#     plt.imshow(right_img)
#     plt.title('Right Image')

#     plt.pause(0.1)  # Display each image for 0.1 seconds

#     if data_ind < len(left_img_list) - 1:
#         plt.clf()  # Clear the current figure for overlaying the next image

# plt.ioff()  # Turn off interactive mode
# plt.show()  # Display final overlayed images

# # %%
# left_depth = read_numpy_file(left_depth_list[data_ind])
# left_depth_vis = depth2vis(left_depth)

# right_depth = read_numpy_file(right_depth_list[data_ind])
# right_depth_vis = depth2vis(right_depth)

# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# plt.imshow(left_depth_vis)
# plt.title('Left Depth')
# plt.subplot(122)
# plt.imshow(right_depth_vis)
# plt.title('Right Depth')
# plt.show()

# # %%
# # Directory containing the locally available files
# local_files_dir = "=dir"

# # Define environment-related functions
# def get_environment_list():
#     '''
#     List all the environments shown in the root directory
#     '''
#     # Implementation depends on how environment information is stored locally
#     # Modify this function accordingly
#     pass

# def get_trajectory_list(envname, easy_hard='Easy'):
#     '''
#     List all the trajectory folders, which are named as 'P0XX'
#     '''
#     # Implementation depends on how trajectory information is stored locally
#     # Modify this function accordingly
#     pass

# def _list_files_in_folder(folder_name):
#     """
#     List all files in a folder
#     """
#     files = []
#     for root, _, filenames in os.walk(folder_name):
#         for filename in filenames:
#             files.append(os.path.join(root, filename))
#     return files

# def get_image_list(trajdir, left_right='left'):
#     assert (left_right == 'left' or left_right == 'right')
#     files = _list_files_in_folder(os.path.join(trajdir, 'image_' + left_right))
#     files = [fn for fn in files if fn.endswith('.png')]
#     return files

# def get_depth_list(trajdir, left_right='left'):
#     assert (left_right == 'left' or left_right == 'right')
#     files = _list_files_in_folder(os.path.join(trajdir, 'depth_' + left_right))
#     files = [fn for fn in files if fn.endswith('.npy')]
#     return files

# def get_flow_list(trajdir):
#     files = _list_files_in_folder(os.path.join(trajdir, 'flow'))
#     files = [fn for fn in files if fn.endswith('flow.npy')]
#     return files

# def get_flow_mask_list(trajdir):
#     files = _list_files_in_folder(os.path.join(trajdir, 'flow'))
#     files = [fn for fn in files if fn.endswith('mask.npy')]
#     return files

# def get_posefile(trajdir, left_right='left'):
#     assert (left_right == 'left' or left_right == 'right')
#     return os.path.join(trajdir, f'pose_{left_right}.txt')

# def get_seg_list(trajdir, left_right='left'):
#     assert (left_right == 'left' or left_right == 'right')
#     files = _list_files_in_folder(os.path.join(trajdir, 'seg_' + left_right))
#     files = [fn for fn in files if fn.endswith('.npy')]
#     return files

# # Function to handle local files
# def handle_local_files():
#     for root, dirs, files in os.walk(local_files_dir):
#         for file in files:
#             # Extracting the file name and extension
#             filename, file_extension = os.path.splitext(file)

#             # If it's a zip file, extract it
#             if file_extension == '.zip':
#                 extract_zip(os.path.join(root, file), extract_to)
#                 print(f"Extracted {file}")

# # Call the function to handle local files
# handle_local_files()

# # %%
# left_seg = read_numpy_file(left_seg_list[data_ind])
# left_seg_vis = seg2vis(left_seg)

# right_seg = read_numpy_file(right_seg_list[data_ind])
# right_seg_vis = seg2vis(right_seg)

# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# plt.imshow(left_seg_vis)
# plt.title('Left Segmentation')
# plt.subplot(122)
# plt.imshow(right_seg_vis)
# plt.title('Right Segmentation')
# plt.show()

# # %%
# flow = read_numpy_file(flow_list[data_ind])
# flow_vis = flow2vis(flow)

# flow_mask = read_numpy_file(flow_mask_list[data_ind])
# flow_vis_w_mask = flow2vis(flow, mask=flow_mask)

# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# plt.imshow(flow_vis)
# plt.title('Optical Flow')
# plt.subplot(122)
# plt.imshow(flow_vis_w_mask)
# plt.title('Optical Flow w/ Mask')
# plt.show()
