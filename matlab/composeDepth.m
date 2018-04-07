function composeDepth( depth_path, normal_path, normal_weight_path, dataset, output_path, param )
%COMPOSEDEPTH Generate complete depth from DNN estimation

depth2depth_path = '../gaps/bin/x86_64/depth2depth';

switch dataset
    case 'realsense'
        if ~exist(output_path,'dir')
            mkdir(output_path);
        end
        test_files = dir([normal_path '/*_normal_est.h5']);
        raw_depth_path_pattern = [depth_path '/%s/%s_depth_open.png'];
        normal_path_pattern = [normal_path '/%s_%s_normal_est.h5'];
        for a = 1:length(test_files)
            tmp = regexp(test_files(a).name, '_', 'split');
            fprintf('>>>>>%d/%d<<<<<<<\n', a, length(test_files));
            
            input_depth_png  = sprintf(raw_depth_path_pattern, tmp{1}, tmp{2});
            input_normal_h5 = sprintf(normal_path_pattern, tmp{1}, tmp{2});
            
            for b = 1:size(param,1)
                output_depth_png = sprintf('%s/%s_%s_%d.png', output_path, tmp{1}, tmp{2}, b);
                cmd = sprintf('%s %s %s -xres %f -yres %f -fx %f -fy %f -cx %f -cy %f -inertia_weight %f -smoothness_weight %f', ...
                    depth2depth_path, input_depth_png, output_depth_png, ...
                    320, 240, 308.331, 308.331, 165.7475, 119.8889, ...
                    param(b,1), param(b,2));
                if ~isempty(normal_path)
                    cmd = sprintf('%s -tangent_weight %f -input_normals %s', cmd, param(b,3), input_normal_h5);
                end       
                if ~isempty(normal_weight_path)
                    cmd = sprintf('%s -input_tangent_weight %s/%s_%s_weight.png', cmd, normal_weight_path, tmp{1}, tmp{2});
                end

                system(cmd);
            end
            
        end
    
    case 'scannet'
        raw_root = depth_path;
        list = '../torch/data_list/scannet_test_list_small.txt';
        
        fp = fopen(list);
        tmp = textscan(fp, '%s');
        test_data_name = tmp{1};
        fclose(fp);
        
        sid = cell(length(test_data_name),1);
        vidc = cell(length(test_data_name),1);
        vidd = cell(length(test_data_name),1);
        for a = 1:length(test_data_name)
            tmp = regexp(test_data_name{a}, '/', 'split');
            sid{a} = tmp{1};
            vidc{a} = tmp{3};
            vidd{a} = strrep(vidc{a}, '_suffix', '');
        end
        
        raw_depth_path_pattern = [raw_root '/%s/depth/%s.png'];
        normal_path_pattern = [result_root normal_path '/%s_%s_normal_est.h5'];
        
        if ~exist(output_path,'dir')
            mkdir(output_path);
        end
        save([output_path '/param.mat'], 'param', 'raw_root', 'normal_path', 'normal_weight_path');
    
        load('scannettestcamera.mat');
        testcamera = scannettestcamera;
        
        for a = 1:length(test_data_name)
            fprintf('>>>>>%d/%d<<<<<<<\n', a, length(test_data_name));
            
            input_depth_png  = sprintf(raw_depth_path_pattern, sid{a}, vidd{a});
            input_normal_h5 = sprintf(normal_path_pattern, sid{a}, vidc{a});

            for b = 1:size(param,1)
                output_depth_png = sprintf('%s/%s_%s_%d.png', output_path, sid{a}, vidc{a}, b);
                cmd = sprintf('%s %s %s -xres %f -yres %f -fx %f -fy %f -cx %f -cy %f -inertia_weight %f -smoothness_weight %f', ...
                    depth2depth_path, input_depth_png, output_depth_png, ...
                    testcamera(b,1), testcamera(b,2), testcamera(b,3), testcamera(b,4), testcamera(b,5), testcamera(b,6), ...
                    param(b,1), param(b,2));
                if ~isempty(normal_path)
                    cmd = sprintf('%s -normal_weight %f -input_normals %s', cmd, param(b,3), input_normal_h5);
                end       
                if ~isempty(normal_weight_path)
                    cmd = sprintf('%s -input_normal_weight %s/%s_%s_weight.png', cmd, normal_weight_path, sid{a}, vidc{a});
                end
                
                system(cmd);
            end
        end
        
    
    case 'mp_render'
        raw_root = depth_path;
        list = '../torch/data_list/mp_test_list_horizontal.txt';
        
        fp = fopen(list);
        tmp = textscan(fp, '%s'); 
        test_data_name = tmp{1};
        fclose(fp);

        sid = cell(length(test_data_name),1);
        vidc = cell(length(test_data_name),1);
        vidd = cell(length(test_data_name),1);
        for a = 1:length(test_data_name)
            tmp = regexp(test_data_name{a}, '/', 'split');
            sid{a} = tmp{1};
            vidc{a} = tmp{3}(1:end-4);
            vidd{a} = strrep(vidc{a}, '_i', '_d');
        end
        
        raw_depth_path_pattern = [raw_root '/%s/undistorted_depth_images/%s.png'];
        normal_path_pattern = [normal_path '/%s_%s_normal_est.h5'];
        
        if ~exist(output_path,'dir')
            mkdir(output_path);
        end
        save([output_path '/param.mat'], 'param', 'depth_path', 'normal_path', 'normal_weight_path');
        
        load('matterporttestcamera.mat');
        testcamera = testcamera;
        for a = 1:length(test_data_name)
            fprintf('>>>>>%d/%d<<<<<<<\n', a, length(test_data_name));
            
            input_depth_png  = sprintf(raw_depth_path_pattern, sid{a}, vidd{a});
            input_normal_h5 = sprintf(normal_path_pattern, sid{a}, vidd{a});
            
            for b = 1:size(param,1)
                output_depth_png = sprintf('%s/%s_%s_%d.png', output_path, sid{a}, vidd{a}, b);
                cmd = sprintf('%s %s %s -xres %f -yres %f -fx %f -fy %f -cx %f -cy %f -inertia_weight %f -smoothness_weight %f', ...
                    depth2depth_path, input_depth_png, output_depth_png, ...
                    testcamera(b,1), testcamera(b,2), testcamera(b,3), testcamera(b,4), testcamera(b,5), testcamera(b,6), ...
                    param(b,1), param(b,2));
                if ~isempty(normal_path)
                    cmd = sprintf('%s -normal_weight %f -input_normals %s', cmd, param(b,3), input_normal_h5);
                end       
                if ~isempty(normal_weight_path)
                    cmd = sprintf('%s -input_normal_weight %s/%s_%s_weight.png', cmd, normal_weight_path, sid{a}, vidd{a});
                end
                
                system(cmd);
            end
            
        end   
    otherwise
        fprintf('Unknown dataset!\n');
end








end

