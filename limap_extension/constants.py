import numpy as np

# yapf: disable
CAM_INTRINSIC = np.array((
    (320,   0, 320),
    (  0, 320, 240),
    (  0,   0,   1)))

H_IMG_TO_CAM = np.array((
    (0, 1, 0, 0),
    (1, 0, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1)
))
# yapf: enable
