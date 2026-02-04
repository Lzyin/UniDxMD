# UniDxMD: Towards Unified Representation for Cross-Modal Unsupervised Domain Adaptation in 3D Semantic Segmentation [ICCV 2025]

## Overview
<img src="asset/framework.png" alt="framework" width="480">

## Preparation

### Installation

The implementation runs on
- Python 3.10
- Torch 1.11.0
- Torchvision 0.12.0
- CUDA 11.4
- [Spatial Sparse Convolution Library (SpConv 2.1)](https://github.com/traveller59/spconv)
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

For the 3D network, we use SpConv, following UniDSeg(https://github.com/Barcaaaa/UniDSeg), as it is faster to compute. SpConv can be installed/compiled as follows:
```
git clone https://github.com/traveller59/spconv, cd ./spconv, pip install -e .
```
### Dataset

NuScenes-Lidarseg:
- Please download the Full dataset from the [NuScenes website](https://www.nuscenes.org/) and extract it.
- Please edit the script UniDxMD/data/nuscenes_lidarseg/preprocess.py as follows and then run it.
  - ```root_dir``` should point to the root directory of the NuScenes dataset.
  - ```out_dir``` should point to the desired output directory to store the pickle files.

A2D2:
- Please download the Semantic Segmentation dataset and Sensor Configuration from the [Audi website](https://www.a2d2.audi/a2d2/en/download.html) or directly use ```wget``` and the following links, then extract.
- Please edit the script xmuda/data/a2d2/preprocess.py as follows and then run it.
  - ```root_dir``` should point to the root directory of the A2D2 dataset.
  - ```out_dir``` should point to the desired output directory to store the undistorted images and pickle files. It should be set differently than the ```root_dir``` to prevent overwriting of images.

SemanticKITTI:
- Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same folder. Similar to NuScenes preprocessing, we save all points that project into the front camera image as well as the segmentation labels to a pickle file.
- Please edit the script xmuda/data/semantic_kitti/preprocess.py as follows and then run it.
  - ```root_dir``` should point to the root directory of the SemanticKITTI dataset.
  - ```out_dir``` should point to the desired output directory to store the pickle files.

VirtualKITTI:
- Clone the repo from [VirtualKITTI website](https://github.com/VisualComputingInstitute/vkitti3D-dataset.git).
- Download raw data and extract with the following script.
  ```
  cd vkitti3D-dataset/tools
  mkdir path/to/virtual_kitti
  bash download_raw_vkitti.sh path/to/virtual_kitti
  ```
- Generate point clouds (npy files).
  ```
  cd vkitti3D-dataset/tools
  for i in 0001 0002 0006 0018 0020; do python create_npy.py --root_path path/to/virtual_kitti --out_path path/to/virtual_kitti/vkitti_npy --sequence $i; done
  ```
- Similar to NuScenes preprocessing, we save all points and segmentation labels to a pickle file.
- Please edit the script `xmuda/data/virtual_kitti/preprocess.py` as follows and then run it.
  - `root_dir` should point to the root directory of the VirtualKITTI dataset.
  - `out_dir` should point to the desired output directory to store the pickle files.
