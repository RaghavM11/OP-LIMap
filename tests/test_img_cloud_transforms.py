import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(REPO_ROOT.as_posix())
from limap_extension.img_cloud_transforms import (imgs_to_clouds_np, cloud_to_img_np, reproject_img,
                                                  xyz_to_uvz, uvz_to_xyz)
from limap_extension.constants import CAM_INTRINSIC
from .test_utils import read_test_data


def test_proj_reproj_same_img():
    """Tests the projection from image to cloud back to image"""
    td_1, _ = read_test_data()

    cloud, corner_idxs = imgs_to_clouds_np(td_1.rgb, td_1.depth, CAM_INTRINSIC)
    img, depth_img, bbox = cloud_to_img_np(cloud, CAM_INTRINSIC)

    # assert np.allclose(td_1.rgb, img, atol=1e-3)
    # Direct comparison is okay since the data type is uint8
    assert (td_1.rgb == img).all()

    assert (td_1.depth == depth_img).all()

    assert bbox.x_min == 0
    assert bbox.x_max == 480
    assert bbox.y_min == 0
    assert bbox.y_max == 640


def test_xyz_uvz_conversion():
    us, vs = np.meshgrid(np.arange(640), np.arange(480))
    us = us.flatten()
    vs = vs.flatten()
    zs = np.ones_like(us)

    xyz = uvz_to_xyz(us, vs, zs)
    us_out, vs_out, zs_out = xyz_to_uvz(xyz)

    assert np.allclose(us, us_out)
    assert np.allclose(vs, vs_out)
    assert np.allclose(zs, zs_out)
