%% Demo for depth completion on RGBD images from Intel Realsense.
% Torch must be installed.
% If command lines cannot be called, please copy and run them in terminal.

cd('../torch/');

% Boundary detection
cmd = 'th main_test_bound_realsense.lua -test_model ../pre_train_model/bound.t7 -test_file ./data_list/realsense_list.txt -root_path ../data/realsense/';
system(cmd); % the result should be in ../torch/result/

% Surface normal estimation
cmd = 'th main_test_realsense.lua -test_model ../pre_train_model/normal_scannet.t7 -test_file ./data_list/realsense_list.txt -root_path ../data/realsense/';
system(cmd);

cd('../matlab/');

% Get occlusion boundary (the 2nd channel) and convert to weight
GenerateOcclusionWeight('../torch/result/bound_realsense_test_bound/', '../torch/result/bound_realsense_weight/');

% Compose depth by global optimization
composeDepth('../data/', '../torch/result/normal_scannet_realsense_test', '../torch/result/bound_realsense_weight', 'realsense', '../results/realsense/', [1000, 0.001, 1]);

% Visualize results
output = imread('../results/realsense/realsense_030_1.png');
colormap = jet(double(max([input(:);output(:)])));
figure;
subplot(1,2,1); imshow(label2rgb(input, colormap)); title('Input');
subplot(1,2,2); imshow(label2rgb(output,colormap)); title('Output');