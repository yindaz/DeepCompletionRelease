function GenerateOcclusionWeight( input_dir, output_dir )
%GENERATEOCCLUSIONWEIGHT Convert the boundary estimation to occlusion
%weight used by optimization
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir([input_dir '/*_bound_est.png']);
for a = 1:length(files)
    fprintf('%d/%d\n',a, length(files));
    bound = imread(fullfile(input_dir, files(a).name));
    bound = im2double(bound);
    weight = (1-bound(:,:,2)).^3;
    imwrite(uint16(weight*1000), fullfile(output_dir, strrep(files(a).name, '_bound_est.png', '_weight.png')));
end

end

