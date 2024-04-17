import numpy as np

# yapf: disable
CAM_INTRINSIC = np.array((
    (320,   0, 320),
    (  0, 320, 240),
    (  0,   0,   1)), dtype=float)

# H_IMG_TO_CAM = np.array((
#     (0, 0, -1, 0),
#     (0, 1, 0, 0),
#     (-1, 0, 0, 0),
#     (0, 0, 0, 1)
# ))
# H_IMG_TO_CAM = np.array((
#     (0, 0, 1, 0),
#     (1, 0, 0, 0),
#     (0, 1, 0, 0),
#     (0, 0, 0, 1)
# ))
H_CAM_TO_NED = np.array((
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (1, 0, 0, 0),
    (0, 0, 0, 1)
), dtype=np.float32)
H_NED_TO_CAM = np.array((
    (0, 0, 1, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 0, 1)
), dtype=np.float32)
# H_IMG_TO_CAM = np.eye(4)
# yapf: enable
