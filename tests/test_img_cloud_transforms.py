import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())
from limap_extension.img_cloud_transforms import (imgs_to_clouds_np, cloud_to_img_np, reproject_img,
                                                  xyz_ned_to_uvz_ocv, uvz_ocv_to_xyz_ned,
                                                  get_uv_coords)
from limap_extension.constants import CAM_INTRINSIC
from tests.test_utils import read_test_data


def test_proj_reproj_same_img():
    """Tests the projection from image to cloud back to image"""
    td_1, _ = read_test_data()

    cloud, corner_idxs = imgs_to_clouds_np(td_1.rgb, td_1.depth, CAM_INTRINSIC)
    img, depth_img, valid_proj_mask, bbox = cloud_to_img_np(cloud, CAM_INTRINSIC)

    # assert np.allclose(td_1.rgb, img, atol=1e-3)
    # Direct comparison is okay since the data type is uint8
    # plt.figure()
    # plt.imshow(td_1.rgb.astype(float) - img.astype(float))
    assert (td_1.rgb == img).all()

    assert (td_1.depth == depth_img).all()

    assert bbox.u_min == 0
    assert bbox.u_max == 639
    assert bbox.v_min == 0
    assert bbox.v_max == 479


# def test_proj_fake_img():
#     """Tests the projection from image to cloud back to image"""
#     rgb_1 = np.zeros((480, 640, 3), dtype=np.uint8)
#     rgb_1[100:102, 100:102, :] = 255
#     depth_1 = np.zeros((480, 640), dtype=np.float32)
#     depth_1[100:102, 100:102] = 1.0

#     rgb_2 = np.zeros((480, 640, 3), dtype=np.uint8)
#     rgb_2[100:102, 100:102, :] = 255
#     depth_2 = np.zeros((480, 640), dtype=np.float32)
#     depth_2[100:102, 100:102] = 3.0

#     pose_1 = np.eye(4)
#     pose_2 = np.eye(4)
#     pose_2[2, 3] = -2.0

#     # cloud, corner_idxs = imgs_to_clouds_np(rgb_1, depth_1, CAM_INTRINSIC)
#     # img, depth_img, bbox = cloud_to_img_np(cloud, CAM_INTRINSIC)

#     img_tformed, depth_tformed, valid_bbox = reproject_img(rgb_1, depth_1, pose_1, pose_2)

#     # # assert np.allclose(td_1.rgb, img, atol=1e-3)
#     # # Direct comparison is okay since the data type is uint8
#     # assert (td_1.rgb == img).all()

#     # assert (td_1.depth == depth_img).all()

#     # assert bbox.x_min == 0
#     # assert bbox.x_max == 480
#     # assert bbox.y_min == 0
#     # assert bbox.y_max == 640


def test_uv_correct_order():
    us, vs = get_uv_coords(480, 640)
    assert us.min() == 0
    assert us.max() == 639
    assert vs.min() == 0
    assert vs.max() == 479
    assert us[0] == 0
    assert us[1] == 1
    assert us[640] == 0
    assert vs[0] == 0
    assert vs[1] == 0
    assert vs[640] == 1


def test_xyz_uvz_conversion():
    us, vs = np.meshgrid(np.arange(640), np.arange(480))
    us = us.flatten()
    vs = vs.flatten()
    zs = np.ones_like(us)

    xyz = uvz_ocv_to_xyz_ned(us, vs, zs)
    us_out, vs_out, zs_out = xyz_ned_to_uvz_ocv(xyz)

    assert np.allclose(us, us_out)
    assert np.allclose(vs, vs_out)
    assert np.allclose(zs, zs_out)


if __name__ == "__main__":
    td_1, _ = read_test_data()

    cloud, corner_idxs = imgs_to_clouds_np(td_1.rgb, td_1.depth, CAM_INTRINSIC)
    img, depth_img, valid_proj_mask, bbox = cloud_to_img_np(cloud, CAM_INTRINSIC)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=[10, 10])
    ax[0].imshow(img)
    ax[1].imshow(td_1.rgb)
    # fig.show()
    fig.savefig("test_img_cloud_transforms.png")
