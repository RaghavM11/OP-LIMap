import numpy as np


class BoundingBox:

    def __init__(self, x_min: int, x_max: int, y_min: int, y_max):
        self.x_min: int = x_min
        self.x_max: int = x_max
        self.y_min: int = y_min
        self.y_max: int = y_max

    @staticmethod
    def from_cloud_corner_idxs(corner_idxs: np.ndarray, uv_coords: np.ndarray, img_height: int,
                               img_width: int):
        # If corner indexes are provided, we can use them to get a tighter bounding box on what
        # parts of image 1 are visible in image 2. This should help the flow out.
        upper_left_idx = corner_idxs[0]
        upper_right_idx = corner_idxs[1]
        lower_left_idx = corner_idxs[2]
        lower_right_idx = corner_idxs[3]

        # yapf: disable
        x_min_bound = np.max([
            uv_coords[0, upper_left_idx],
            uv_coords[0, upper_right_idx],
            0
        ])
        x_max_bound = np.min([
            uv_coords[0, lower_left_idx],
            uv_coords[0, lower_right_idx],
            img_height
        ])
        y_min_bound = np.max([
            uv_coords[1, upper_left_idx],
            uv_coords[1, lower_left_idx],
            0
        ])
        y_max_bound = np.min([
            uv_coords[1, upper_right_idx],
            uv_coords[1, lower_right_idx],
            img_width
        ])
        # yapf: enable

        valid_bbox = BoundingBox(x_min_bound, x_max_bound, y_min_bound, y_max_bound)
        return valid_bbox

    def crop_img(self, img: np.ndarray) -> np.ndarray:
        cropped = None
        if img.ndim == 2:
            cropped = img[self.x_min:self.x_max, self.y_min:self.y_max]
        elif img.ndim == 3:
            if img.shape[0] == 3:
                raise ValueError("Expected image to be in HWC format, got CHW instead.")
            cropped = img[self.x_min:self.x_max, self.y_min:self.y_max, :]
        else:
            raise ValueError("Image must be 2D or 3D")
        return cropped
