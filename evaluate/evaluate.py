import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

SEG_DIR = # OS PATHS HERE: '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P007/seg_left/'
GROUND_TRUTH_DIR = # OS PATHS HERE'/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/P007/ground_truth_mask/'

def read_seg_mask():

    for seg in os.listdir(SEG_DIR):
        seg = os.path.join(SEG_DIR, seg)
        seg_mask = np.load(seg)
        dynamic_mask = np.zeros_like(seg_mask)
        dynamic_mask[seg_mask == 232] = 1
        # now save dynamic mask in the ground thruth directory
        ground_truth_mask = os.path.join(GROUND_TRUTH_DIR, seg.split('/')[-1])
        if not os.path.exists(GROUND_TRUTH_DIR):
            os.makedirs(GROUND_TRUTH_DIR)
        np.save(ground_truth_mask, dynamic_mask)

def read_dyn_mask():
    for dyn in os.listdir(GROUND_TRUTH_DIR):
        dyn = os.path.join(GROUND_TRUTH_DIR, dyn)
        dyn_mask = np.load(dyn)

def check_intersection(x1, y1, x2, y2, ix, iy):
    if min(x1, x2) <= ix <= max(x1, x2) and min(y1, y2) <= iy <= max(y1, y2):
        line_equation = lambda x: (y2 - y1) / (x2 - x1) * (x - x1) + y1
        if abs(line_equation(ix) - iy) < 1e-6:
            return True
    return False
     

read_seg_mask()
read_dyn_mask()

# def get_2d_lines(img):
#     """Get 2D lines from an image."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
#     return lines # returns r and theta

# def distance_between_lines(line1_start, line1_end, line2_start, line2_end):
#     # Vector representing line1
#     v1 = line1_end - line1_start
#     # Vector representing line2
#     v2 = line2_end - line2_start
#     # Vector from line1 start to line2 start
#     v3 = line2_start - line1_start
    
#     # Compute dot products
#     dot11 = np.dot(v1, v1)
#     dot12 = np.dot(v1, v2)
#     dot13 = np.dot(v1, v3)
#     dot22 = np.dot(v2, v2)
#     dot23 = np.dot(v2, v3)
    
#     # Compute parameters of closest points on each line
#     denominator = dot11 * dot22 - dot12 ** 2
#     if denominator != 0:
#         s = (dot12 * dot23 - dot22 * dot13) / denominator
#         t = (dot11 * dot23 - dot12 * dot13) / denominator
        
#         # Check if the closest points are within the line segments
#         if 0 <= s <= 1 and 0 <= t <= 1:
#             # Compute the closest points
#             closest_point_line1 = line1_start + s * v1
#             closest_point_line2 = line2_start + t * v2
#             # Compute the distance between the closest points
#             distance = np.linalg.norm(closest_point_line1 - closest_point_line2)
#             return distance
#     # If the lines are parallel, compute the distance between start points
#     return np.linalg.norm(line1_start - line2_start)


# point1 = np.array([0, 0])
# point2 = np.array([1, 0])

# point3 = np.arrya2, 1
# x3, y3 = 3, 1

