% Realsense raw depth can be very noisy. Depth error on sparse pixels can
% be very wrong. Remove signals on boundary.

root_dir = './realsense/';
fp = fopen('../torch/data_list/realsense_list.txt');
temp = textscan(fp, '%s');
data_list = temp{1};
fclose(fp);

for a = 1:length(data_list)
    fprintf('%d\n', a);
    
    depth = imread([data_list{a} '_depth.png']);
    
    valid = depth>0;
    dist = bwdist(~valid);     
    depth(dist<1.5) = 0;
    depth = imdilate(depth, ones(3,3));
    imwrite(depth, [data_list{a} '_depth_open.png']);
end