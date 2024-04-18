import numpy as np


class CornerIdxs:

    def __init__(self, upper_left: int, upper_right: int, lower_left: int, lower_right: int):
        self.upper_left = upper_left
        self.upper_right = upper_right
        self.lower_left = lower_left
        self.lower_right = lower_right

    def as_np_array(self) -> np.ndarray:
        return np.array([self.upper_left, self.upper_right, self.lower_left, self.lower_right])

    def adjust_coords_given_depth_validity(self, depth_valid_mask: np.ndarray, img_cols: int):
        # As there's no guarantee that the corner points have a valid associated depth, we adjust the
        # corner indexes until a valid point is found.
        while depth_valid_mask[self.upper_left] is False:
            self.upper_left += 1

        while depth_valid_mask[self.upper_right] is False:
            self.upper_right += img_cols

        while depth_valid_mask[self.lower_left] is False:
            self.lower_left += 1

        while depth_valid_mask[self.lower_right] is False:
            self.lower_right -= img_cols


class BoundingBox:

    def __init__(self, u_min: int, u_max: int, v_min: int, v_max):
        self.u_min: int = u_min
        self.u_max: int = u_max
        self.v_min: int = v_min
        self.v_max: int = v_max

    @staticmethod
    def from_cloud_corner_idxs(corner_idxs: CornerIdxs, uv_coords: np.ndarray, img_height: int,
                               img_width: int):
        # If corner indexes are provided, we can use them to get a tighter bounding box on what
        # parts of image 1 are visible in image 2. This should help the flow out.
        # upper_left_idx = corner_idxs[0]
        # upper_right_idx = corner_idxs[1]
        # lower_left_idx = corner_idxs[2]
        # lower_right_idx = corner_idxs[3]

        # yapf: disable
        u_min_bound = np.max([
            uv_coords[0, corner_idxs.upper_left],
            uv_coords[0, corner_idxs.lower_left],
            0
        ])
        u_max_bound = np.min([
            uv_coords[0, corner_idxs.upper_right],
            uv_coords[0, corner_idxs.lower_right],
            img_width - 1
        ])
        v_min_bound = np.max([
            uv_coords[1, corner_idxs.upper_left],
            uv_coords[1, corner_idxs.upper_right],
            0
        ])
        v_max_bound = np.min([
            uv_coords[1, corner_idxs.lower_left],
            uv_coords[1, corner_idxs.lower_right],
            img_height - 1
        ])
        # yapf: enable

        valid_bbox = BoundingBox(u_min_bound, u_max_bound, v_min_bound, v_max_bound)
        return valid_bbox

    def crop_img(self, img: np.ndarray) -> np.ndarray:
        cropped = None
        if img.ndim == 2:
            cropped = img[self.v_min:self.v_max, self.u_min:self.u_max]
        elif img.ndim == 3:
            if img.shape[0] == 3:
                raise ValueError("Expected image to be in HWC format, got CHW instead.")
            cropped = img[self.v_min:self.v_max, self.u_min:self.u_max, :]
        else:
            raise ValueError("Image must be 2D or 3D")
        return cropped

    def uncrop_img(self, img: np.ndarray, img_height_orig: int, img_width_orig: int, fill_value):
        if img.ndim == 2:
            uncropped = np.ones((img_height_orig, img_width_orig), dtype=img.dtype) * fill_value
            uncropped[self.v_min:self.v_max, self.u_min:self.u_max] = img
        elif img.ndim == 3:
            if img.shape[0] == 3:
                raise ValueError("Expected image to be in HWC format, got CHW instead.")
            uncropped = np.ones(
                (img_height_orig, img_width_orig, img.shape[2]), dtype=img.dtype) * fill_value
            uncropped[self.v_min:self.v_max, self.u_min:self.u_max, :] = img
        else:
            raise ValueError("Image must be 2D or 3D")

        return uncropped

    def __str__(self):
        return f"BBox: u_min={self.u_min}, u_max={self.u_max}, v_min={self.v_min}, v_max={self.v_max}"
