from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path
    import sys
    REPO_DIR = Path(__file__).resolve().parents[2]
    sys.path.append(REPO_DIR.as_posix())
    from limap_extension.bounding_box import BoundingBox

def display_img_pair(rgb, depth, img_slice: 'BoundingBox' = None):
    if img_slice is not None:
        rgb = img_slice.crop_img(rgb)
        depth = img_slice.crop_img(depth)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)

    # depth = np.clip(depth, 0, 10)
    im1 = ax[1].imshow(depth)
    fig.colorbar(im1, ax=ax[1])