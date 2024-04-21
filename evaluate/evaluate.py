import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


SEG_DIR = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/datasets/carwelding/Hard/P001/seg_left' 
FINALTRACKS_SAMPLE_PATH = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/triangulation/finaltracks/'

GROUND_TRUTH_DIR_1 = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/P001/ground_truth_mask/'
GROUND_TRUTH_DIR_2 = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/P002/ground_truth_mask/'
GROUND_TRUTH_DIR_3 = '/Users/shlokagarwal/Desktop/Mobile Robotics/project/LIMap-Extension/P003/ground_truth_mask/'

# SEG_DIR = '/home/saketp/Desktop/LIMap-Extension/datasets/carwelding_Hard_P001_with_flow_masks/carwelding/Hard/P001/seg_left' 
# FINALTRACKS_SAMPLE_PATH = '/home/saketp/Desktop/LIMap-Extension/tests/data/finaltracks_sample/'

# GROUND_TRUTH_DIR_1 = '/home/saketp/Desktop/LIMap-Extension/datasets/carwelding_Hard_P001_with_flow_masks/carwelding/Hard/P001/ground_truth_mask/'
# GROUND_TRUTH_DIR_2 = '/home/saketp/Desktop/LIMap-Extension/datasets/carwelding_Hard_P002_with_flow_masks/carwelding/Hard/P002/ground_truth_mask/'
# GROUND_TRUTH_DIR_3 = '/home/saketp/Desktop/LIMap-Extension/datasets/carwelding_Hard_P003_with_flow_masks/carwelding/Hard/P003/ground_truth_mask/'


def read_seg_mask():
    for seg in os.listdir(SEG_DIR):
        seg = os.path.join(SEG_DIR, seg)
        seg_mask = np.load(seg)
        dynamic_mask = np.zeros_like(seg_mask)
        dynamic_mask[seg_mask == 232] = 1

        # now save dynamic mask in the ground truth directory
        ground_truth_mask = os.path.join(GROUND_TRUTH_DIR_3, seg.split('/')[-1])
        if not os.path.exists(GROUND_TRUTH_DIR_3): 
            os.makedirs(GROUND_TRUTH_DIR_3)
        np.save(ground_truth_mask, dynamic_mask)
    

def read_dyn_mask():
    for dyn in os.listdir(GROUND_TRUTH_DIR_3):
        if not dyn.startswith("track"): continue
        dyn = os.path.join(GROUND_TRUTH_DIR_3, dyn)
        dyn_mask = np.load(dyn)


def check_intersection(x1, y1, x2, y2, ix, iy):
    if x1 == x2: return abs(x1 - ix) < 1e-6
    if min(x1, x2) <= ix <= max(x1, x2) and min(y1, y2) <= iy <= max(y1, y2):
        line_equation = lambda x: (y2 - y1) / (x2 - x1) * (x - x1) + y1
        if abs(line_equation(ix) - iy) < 1e-6: return True
    return False


def check_intersection_lines(lines, points):
    points = np.array(points)
    lines = np.array(lines)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    x, y = points[:, 0], points[:, 1]
    
    m_lines = (y2 - y1) / (x2 - x1)
    b_lines = y1 - m_lines * x1

    vertical_lines = np.isinf(m_lines)
    vertical_x = x1[vertical_lines]
    
    x_intersections = (y - b_lines[:, np.newaxis]) / m_lines[:, np.newaxis]
    x_intersections[vertical_lines] = vertical_x[:, np.newaxis]
    
    intersection_within_x = np.logical_and(x >= np.minimum(x1, x2)[:, np.newaxis], 
                                           x <= np.maximum(x1, x2)[:, np.newaxis])
    intersection_within_y = np.logical_and(y >= np.minimum(y1, y2)[:, np.newaxis], 
                                           y <= np.maximum(y1, y2)[:, np.newaxis])
    
    intersection_within_x = np.logical_and(abs(x - x_intersections) < 1e-6, intersection_within_x)
    intersection_mask = np.logical_and(intersection_within_x, intersection_within_y)
    any_intersection = np.any(intersection_mask, axis=1)

    return any_intersection


image_id_arrays = {}
line2d_arrays = {}

for filename in os.listdir(FINALTRACKS_SAMPLE_PATH):
    if filename.startswith("track_") and filename.endswith(".txt"):
        with open(os.path.join(FINALTRACKS_SAMPLE_PATH, filename), 'r') as file: lines = file.readlines()

        image_id_list = []
        line2d_list = []

        for line_index, line in enumerate(lines):
            if line.startswith("image_id_list"): image_id_list = line.split()[2:]  
            elif line.startswith("line2d_list"):
                end_index = len(lines)
                for end_line_index, end_line in enumerate(lines[line_index + 1:]):
                    if end_line.startswith("node_id_list"):
                        end_index = line_index + 1 + end_line_index
                        break
                
                line2d_list = [list(map(float, point.split())) for point in lines[line_index + 1:end_index]]
                line2d_array = np.array(line2d_list)

        image_id_array = np.array(image_id_list, dtype=int)
        image_id_arrays[filename] = image_id_array
        line2d_arrays[filename] = line2d_array


for filename in image_id_arrays.keys():
    print(f"Track: {filename}")
    print("Image ID List:")
    print(image_id_arrays[filename])
    print("Line2D List:")
    print(line2d_arrays[filename])


# Uncomment the 2 lines below to get the segmentation and dynamic masks; store them in your respective directories

read_seg_mask()
read_dyn_mask()


dyn_mask_list = {}

for filename in os.listdir(GROUND_TRUTH_DIR_3):
    if filename.endswith('.npy'):
        print(filename)
        filepath = os.path.join(GROUND_TRUTH_DIR_3, filename)
        data = np.load(filepath)
        scene_id = int(filename.split('_')[0])
        print (scene_id)
        dyn_mask_list[scene_id] = data
output_file = 'linetrack_scores_3.txt'
cum_score = 0
total_lines = 0
with open(output_file, 'w') as file:
    for linetrack, lines in line2d_arrays.items():
        scene_ids = image_id_arrays[linetrack] 
        score = 0
        # print("just checking")
        # print(lines, scene_ids, linetrack)
        total_lines += len(lines)
        for scene_id in range(len(scene_ids)):
            if scene_id in dyn_mask_list: 
                dyn_mask = dyn_mask_list[scene_id]
                li = lines[scene_id]
                li = np.reshape(li, (1, 4))
                locs = np.where(dyn_mask == 1)

                if locs[0].size > 0:
                    intersection = check_intersection_lines(li, locs)
                    if not intersection[0]:
                        score += 1
                    # print(intersection)
        #             if not np.any(intersection): score += 1
        #     else: print(f"Dynamic mask not found for scene_id {scene_id}")
        cum_score += score
        # print(f"Score for linetrack {linetrack}: {score}")
        file.write(f"Score for linetrack {linetrack}: {score}\n")
    file.write(f"Total lines (2d projections of the track): {total_lines}\n")
    file.write(f"Total score: {cum_score}\n")

print(f"Scores saved to {output_file}")

# all_pixel_locations = []
# for scene_id, dyn_mask in dyn_mask_list.items():
#     pixel_locations = np.where(dyn_mask == 1)
#     all_pixel_locations.extend(list(zip(pixel_locations[0], pixel_locations[1])))
#     # print(f"Pixel locations with value 1 in {filename}:")
#     # for x, y in zip(pixel_locations[0], pixel_locations[1]):
#     #     print(f"({x}, {y})")


# all_locations_array = np.array(all_pixel_locations)
# print("All pixel locations with value 1 across all images:")
# print(all_locations_array)


# for filename, line2d_array in line2d_arrays.items():
#     score_counts[filename] = []
#     print('done')
#     for ix, iy in all_locations_array:
#         score = 0
#         count = 0
#         for line_segment in line2d_array:
#             x1, y1, x2, y2 = line_segment
#             if check_intersection(x1, y1, x2, y2, ix, iy): score += 1
#             count += 1
#         score_counts[filename].append((ix, iy, score, count))


# for filename, scores in score_counts.items():
#     print(f"Track: {filename}")
#     print("Scores:")
#     for score in scores: print(score)






# Extra code 

# Sample block for check_intersection()
# x1, y1 = 1, 1
# x2, y2 = 5, 5
# x_point, y_point = 3, 3
# print(check_intersection(x1, y1, x2, y2, x_point, y_point))


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