# LIMap-Extension
#mobilerobotics project

# Installation

## Cloning The Repo

This repository has multiple nested Git `submodules`, so the cloning procedure is *slightly* more complex than usual. To clone the entirety of the repository:
```
git clone https://github.com/RaghavM11/LIMap-Extension.git
cd LIMap-Extension/
git submodule update --init --recursive
```

# For running examples
## To run line mapping using Fitnmerge (line mapping with available depth maps) on Hypersim:
```
python limap/runners/hypersim/fitnmerge.py --output_dir limap/outputs/quickstart_fitnmerge
```

## To run Visualization of the 3D line maps after the reconstruction:

```
python limap/visualize_3d_lines.py --input_dir outputs/quickstart_triangulation/finaltracks \
                             # add the camera frustums with "--imagecols outputs/quickstart_triangulation/imagecols.npy"
```

## For Hybrid Loaclization of Poinys and Lines using NN detectors

```
python limap/runners/tests/localization.py --data limap/runners/tests/data/localization/localization_test_data_stairs_1.npy
```