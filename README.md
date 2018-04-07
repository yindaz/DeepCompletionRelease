# Deep Depth Completion of a Single RGB-D Image

## Introduction:

This repository includes code/data described in paper:

@article{zhang2018deepdepth,
	  title={Deep Depth Completion of a Single RGB-D Image},
	  author={Zhang, Yinda and Funkhouser, Thomas},
	  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2018}
	}

Please check the project webpage (http://deepcompletion.cs.princeton.edu/) for more details. If you have any question, please contact Yinda Zhang (yindaz guess cs dot princeton dot edu).

## Quick testing
1. Download realsense data in [`./data/`](./data), and unzip it there.
2. Download `bound.t7` and `normal_scannet.t7` in [`./pre_train_model/`](./pre_train_model).
3. Compile `depth2depth` in [`./gaps/`](./gaps).
4. Run `demo_realsense.m` in [`./matlab/`](./matlab).

## Pre-trained Model
We provide pre-trained models on Matterport3D and ScanNet datasets. Please check [`./pre_train_model/`](./pre_train_model) for download links.

## Data
We collect our training and testing data from SUNCG, Matterport3D, and ScanNet dataset. Each dataset has its individual license/agreement, which requires users to sign an agreement form before using the data. To access our rendered ground truth, please send an email to the dataset orgnizer to confirm your agreement and cc Yinda Zhang (yindaz guess cs dot princeton dot edu). You will get download link for the data from the dataset that you got approval. For a list of data we provide, please check [`./data/`](./data)

- SUNCG-RGBD. We use high quality synthetic data to train the occlusion detection network. Please check http://pbrs.cs.princeton.edu/ for details of the dataset and how to sign agreement.
- Matterport3D. We render depth from Matterport3D dataset for training normal estimation model and evaluating depth completion. Please check https://github.com/niessner/Matterport for details of the dataset and how to sign agreement.
- ScanNet. We render depth from ScanNet dataset for training normal estimation model and evaluating depth completion. Please check https://github.com/ScanNet/ScanNet for details of the dataset and how to sign agreement.


## Occlusion & Surface Normal
We extend codes here: https://github.com/yindaz/surface_normal for occlusion detection and surface normal estimation. Both training and test codes are provided. Please see [`./torch/`](./torch) for more details.

## Optimization
We extend gaps (https://github.com/tomfunkhouser/gaps) for global optimization. Please check [`./gaps/`](./gaps) for the optimization tool.

## Evaluation
- We use the MATLAB code [`./matlab/evalDepth.m`](./matlab/evalDepth.m) to evaluate the performance of the depth completion. 
- We also provide pre-computed results of our method on Matterport3D and ScanNet. Please check [`./torch/data_list/`](./torch/data_list) for the list of testing images. Since the testing data also comes from original dataset, please email Yinda Zhang for download link after getting access to these datasets.