# Deep Depth Completion of a Single RGB-D Image

This repository includes code/data described in this paper:

Yinda Zhang, Thomas Funkhouser  
Deep Depth Completion of a Single RGB-D Image  
Computer Vision and Pattern Recognition (CVPR2018)

	@article{zhang2018deepdepth,
		  title={Deep Depth Completion of a Single RGB-D Image},
		  author={Zhang, Yinda and Funkhouser, Thomas},
		  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		  year={2018}
		}

Please check the project webpage (http://deepcompletion.cs.princeton.edu/) for more details. If you have any question, please contact Yinda Zhang (yindaz guess cs dot princeton dot edu).

## Quick Test
1. Download realsense data in [`./data/`](./data), and unzip it there.
2. Download `bound.t7` and `normal_scannet.t7` in [`./pre_train_model/`](./pre_train_model).
3. Compile `depth2depth` in [`./gaps/`](./gaps).
4. Run `demo_realsense.m` in [`./matlab/`](./matlab).

## Details of Pipeline

Our method completes a depth image using information extracted from an aligned color image. We first estimate surface normal and occlusion boundary from the color image. The surface normal tells relations of depth between nearby pixels, and occlusion boundary indicates depth discontinuity. A global optimization integrates these two information with the observed depth to generate a complete depth image. Implementations for major steps are as the following:
1. Estimating Surface Normal. We train a fully convolutional neural network to estimate surface normal. Training and testing codes can be found in `./torch/main_train_[matterport/scannet].lua` and `./torch/main_test_[matterport/scannet].lua`. The estimated surface normal is saved as a H x W x 3 matrix in HDF5 format.
2. Detecting Boundary. We train a fully convolutional neural network to estimate boundaries. Training and testing codes can be found in `./torch/main_train_bound.lua` and `./torch/main_test_bound_[matterport/scannet].lua`. This network performs a per-pixel classification with 3 labels: 0 for no boundary, 1 for occlusion boundary (depth discontinuity), and 2 for crease (normal discontinuity). The result is a H x W x 3 matrix containing the probability of each pixel belonging to each label saved in png format.
3. Getting Occlusion Weight. The surface normal constraint for pixels on occlusion boundary should be weaken. Run `./matlab/GenerateOcclusionWeight.m` to generate the occlusion based weight map for optimization.
4. Global Optimization. We compute depth from normals using a C++ program called `depth2depth`. Source code for the `depth2depth` program and all its dependencies can be found in the [`./gaps/`](./gaps) directory. Compiling with `cd ./gaps; make` will create an executable in `./gaps/bin/x86_64/depth2depth`. An example command to use `depth2depth` is like:
```
depth2depth input_depth.png output_depth.png -xres 320 -yres 240 -fx 308 -fy 308 -cx 165 -cy 119 -inertia_weight 1000 -smoothness_weight 0.001 -tangent_weight 1 -input_normals PATH_to_Normal_Est.h5 -input_tangent_weight PATH_to_Occlusion_Weight.png
```
- input_depth.png: The path for the raw depth map from sensor, which is the depth to refine. It should be saved as 4000 x depth in meter in a 16bit PNG.
- output_depth.png: The path for the result, which is the completed depth. It is also saved as 4000 x depth in meter in a 16bit PNG.
- xres, fx, cx: The resolution, focal length, and camera center location along horizonal direction. Similar for yres, fy, and cy.
- inertia_weight: The strength of the penalty on the difference between the input and the output depth map on observed pixels. Set this value higher if you want to maintain the observed depth from `input_depth.png`.
- smoothness_weight: The strength of the penalty on the difference between the depths of neighboring pixels. Higher smoothness weight will produce soap-film-like result.
- tangent_weight: The universal strength of the surface normal constraint. Higher tangent weight will force the output to have the same surface normal with the given one.
- input_normals: The estimated surface normal from Step 1.
- input_tangent_weight: The pixel-wised strength of the surface normal constraint, which is the occlusion based weight from Step 3. The final weight on the surface normal constraint is the multiplication of the weight map with the tangent_weight.


## Pre-trained Model

We provide pre-trained models on SUNCG-RGBD, Matterport3D, and ScanNet datasets. Please check [`./pre_train_model/`](./pre_train_model) for download links.

## Depth Completion Dataset

We provide training and testing data for occlusion detection and surface normal estimation. For occlusion detection, we use synthetic dataset SUNCG-RGBD and generate depth based boundary map. For surface normal estimation, we render depth and normal map from reconstructed mesh models from Matterport3D and ScanNet. For a full list of data we provide, please check [`./data/`](./data)

If you are interested in using any part of our data, you must obtain the access to the corresponding original dataset that they are generated from. Unfortunately, SUNCG-RGBG, Matterport3D, and ScanNet are using different terms of usage, and hence you may need to get accesses to all of them in order to have a full access to our dataset. Please send an email to the dataset organizer(s) to confirm your agreement and cc Yinda Zhang (yindaz guess cs dot princeton dot edu). You will get download links for the data from the dataset that you got approval. 

- SUNCG-RGBD. We use high quality synthetic data to train the occlusion detection network. Please check http://pbrs.cs.princeton.edu/ for details of the dataset and how to sign agreement.
- Matterport3D. We render depth from Matterport3D dataset for training normal estimation model and evaluating depth completion. Please check https://github.com/niessner/Matterport for details of the dataset and how to sign agreement.
- ScanNet. We render depth from ScanNet dataset for training normal estimation model and evaluating depth completion. Please check https://github.com/ScanNet/ScanNet for details of the dataset and how to sign agreement.


## Occlusion & Surface Normal
We extend codes here: https://github.com/yindaz/surface_normal for occlusion detection and surface normal estimation. Both training and test codes are provided. Please see [`./torch/`](./torch) for more details.

## Optimization
We extend gaps (https://github.com/tomfunkhouser/gaps) for global optimization. Please check [`./gaps/`](./gaps) for the optimization tool `depth2depth`.

## Evaluation
- We use the MATLAB code [`./matlab/evalDepth.m`](./matlab/evalDepth.m) to evaluate the performance of the depth completion. 
- To evaluate on Matterport3D and/or ScanNet, you need to get accesses to these dataset to download ground truth for the testing set. Specifically, you will need two MAT files containing depth and camera parameters to run optimization and/or evaluation.
- We also provide pre-computed results of our method on Matterport3D and ScanNet. Please check [`./torch/data_list/`](./torch/data_list) for the list of testing images. Since the testing data also comes from original dataset, please email Yinda Zhang for download link after getting access to Matterport3D and/or ScanNet.
