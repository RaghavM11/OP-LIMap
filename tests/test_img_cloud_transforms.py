import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(REPO_ROOT.as_posix())
from scripts.img_cloud_transforms import imgs_to_clouds_np, cloud_to_img_np
from scripts.constants import CAM_INTRINSIC
from .test_utils import read_test_data


def test_proj_reproj():
    """Tests the projection from image to cloud back to image"""
    td_1, _ = read_test_data()

    cloud = imgs_to_clouds_np(td_1.rgb, td_1.depth, CAM_INTRINSIC)
    img = cloud_to_img_np(cloud, CAM_INTRINSIC)

    # assert np.allclose(td_1.rgb, img, atol=1e-3)
    # Direct comparison is okay since the data type is uint8
    assert (td_1.rgb == img).all()
