# UniDxMD: Towards Unified Representation for Cross-Modal Unsupervised Domain Adaptation in 3D Semantic Segmentation [ICCV 2025]

## Overview
<img src="asset/framework.png" alt="framework" width="480">

## Preparation

### Installation

The implementation runs on
- Python 3.10.13
- Torch 1.11.0
- Torchvision 0.12.0
- CUDA 11.4
- [Spatial Sparse Convolution Library (SpConv 2.1)](https://github.com/traveller59/spconv)
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

For the 3D network, we use SpConv, following [UniDSeg](https://github.com/Barcaaaa/UniDSeg), as it is faster to compute. SpConv can be installed/compiled as follows:
```
git clone https://github.com/traveller59/spconv, cd ./spconv, pip install -e .
```

### Dataset

#### NuScenes-Lidarseg:
- Please download the Full dataset from the [NuScenes website](https://www.nuscenes.org/) and extract it.
- Please edit the script UniDxMD/data/nuscenes_lidarseg/preprocess.py as follows and then run it.
  - ```root_dir``` should point to the root directory of the NuScenes-Lidarseg dataset.
  - ```out_dir``` should point to the desired output directory to store the pickle files.

#### VirtualKITTI:
- Clone the following repo:
```
$ git clone https://github.com/VisualComputingInstitute/vkitti3D-dataset.git
```
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
- Similar to NuScenes-Lidarseg preprocessing, we save all points and segmentation labels to a pickle file.
- Please edit the script `UniDxMD/data/virtual_kitti/preprocess.py` as follows and then run it.
  - `root_dir` should point to the root directory of the VirtualKITTI dataset.
  - `out_dir` should point to the desired output directory to store the pickle files.

#### A2D2:
- Please download the Semantic Segmentation dataset and Sensor Configuration from the [Audi website](https://www.a2d2.audi/a2d2/en/download.html) or directly use ```wget``` and the following links, then extract.
```
$ wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar
$ wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/cams_lidars.json
```
For preprocessing, we undistort the images and store them separately as .png files.
Similar to NuScenes-Lidarseg preprocessing, we save all points that project into the front camera image as well
as the segmentation labels to a pickle file.
- Please edit the script UniDxMD/data/a2d2/preprocess.py as follows and then run it.
  - ```root_dir``` should point to the root directory of the A2D2 dataset.
  - ```out_dir``` should point to the desired output directory to store the undistorted images and pickle files. It should be set differently than the ```root_dir``` to prevent overwriting of images.

#### SemanticKITTI:
- Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same folder.
- Similar to NuScenes-Lidarseg preprocessing, we save all points that project into the front camera image as well as the segmentation labels to a pickle file.
- Please edit the script UniDxMD/data/semantic_kitti/preprocess.py as follows and then run it.
  - ```root_dir``` should point to the root directory of the SemanticKITTI dataset.
  - ```out_dir``` should point to the desired output directory to store the pickle files.

## Cross-Modal UDA Experiments

You can run the training with
```
$ cd <root dir of this repo>
$ python UniDxMD/train_UniDxMD.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/UniDxMD.yaml
```

You can start the trainings on the other UDA scenarios (nuScenes: Day/Night, v.KITTI/Sem.KITTI and A2D2/Sem.KITTI) analogously:
```
$ python UniDxMD/train_UniDxMD.py --cfg=configs/nuscenes_lidarseg/day_night/uda/UniDxMD.yaml
$ python UniDxMD/train_UniDxMD.py --cfg=configs/virtual_kitti_semantic_kitti/uda/UniDxMD.yaml
$ python UniDxMD/train_UniDxMD.py --cfg=configs/a2d2_semantic_kitti/uda/UniDxMD.yaml
```
## Testing
You can provide which checkpoints you want to use for testing. We used the ones
that performed best on the validation set during training (the best val iteration for 2D and 3D is
shown at the end of each training). Note that `@` will be replaced
by the output directory for that config file. For example:
```
$ cd <root dir of this repo>
$ python UniDxMD/test.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/UniDxMD.yaml @/best_model_2d.pth @/best_model_3d.pth
```
## Acknowledgements

Code is built based on [xMUDA_journal](https://github.com/valeoai/xmuda_journal) and [UniDSeg](https://github.com/Barcaaaa/UniDSeg).

## Citation

If you find this code useful for your research, please consider citing our paper:
```bibtex
@inproceedings{liang2025unidxmd,
  title={UniDxMD: Towards Unified Representation for Cross-Modal Unsupervised Domain Adaptation in 3D Semantic Segmentation},
  author={Liang, Zhengyin and Yin, Hui and Liang, Min and Du, Qianqian and Yang, Ying and Huang, Hua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20346--20356},
  year={2025}
}
```
