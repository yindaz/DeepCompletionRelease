# Torch implementation of occlusion detection and surface normal estimation

## Data lists
The training and testing data lists can be found in `./data_list/`. Specifically:
- `sync_*` are for occlusion detection.
- `mp_*` are for normal estimation on Matterport3D.
- `scannet_*` are for normal estimation on ScanNet.

## Occlusion detection

### Testing
Run
```
th main_test_bound_realsense.lua -test_model ../pre_train_model/bound.t7 -test_file ./data_list/realsense_list.txt -root_path ../data/realsense/
```

The result will be generated in `./result/bound_realsense_test_bound/`. On ScanNet and Matterport3d, you may use `main_test_bound_[scannet/matterport].lus` to test on `./data_list/scannet_test_list_small.txt` and `./data_list/mp_test_list_horizontal.txt`.

### Training
Run
```
th main_train_bound.lua -pretrain_file ../pre_train_model/sync.t7 -root_path ../data/pbrs_boundary/ -ps ./model/bound
```

The snapshots and models will be saved under `./model/`.

## Surface normal estimation

### Testing
Run
```
th main_test_realsense.lua -test_model ../pre_train_model/normal_scannet.t7 -test_file ./data_list/realsense_list.txt -root_path ../data/realsense/
```

The result will be generated in `./result/normal_scannet_realsense_test/`. On ScanNet and Matterport3d, you may use `main_test_[scannet/matterport].lus` to test on `./data_list/scannet_test_list_small.txt` and `./data_list/mp_test_list_horizontal.txt`.

### Training
Run
```
 th main_train_matterport.lua -ps ./model/normal_matterport -use_render_normal_gt -root_path ../data/to_matterport/
```

The snapshots and models will be saved under `./model/`. Noted that you need to download color image from official Matterport3D dataset in order to train the model. Same as before, you can use `main_train_scannet.lua` to train on ScanNet.
