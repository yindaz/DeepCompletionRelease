function [ metric, example_error ] = evalDepth( result_folder, dataset, suffix )
%EVALDEPTH Evaluate completed depth

switch dataset
    
    case 'scannet'
        list = '../torch/data_list/scannet_test_list_small.txt';
        load('scannettestdata.mat');
        testdata = scannettestdata;
        
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
        

        rs_list = cell(length(test_data_name),1);
        for a = 1:length(test_data_name)
            rs_list{a} = sprintf('%s/%s_%s_%s.png', result_folder, sid{a}, vidc{a}, suffix);  
        end
    
        example_error = repmat(struct('error',[]), length(test_data_name), 1);
        for a = 1:length(test_data_name)
            fprintf('%d/%d\n', a, length(test_data_name));
            result = imread(rs_list{a});
            result = double(result)/4000;
            raw = testdata(a).raw;
            gdt = testdata(a).gdt;
            
            diff = abs(result-gdt);
            
            valid1 = gdt>0;
            valid2 = gdt>0 & raw>0;
            valid3 = gdt>0 & raw==0;
            
            example_error(a).sqr = diff(valid1(:)).^2;
            example_error(a).rel = diff(valid1(:))./gdt(valid1(:));
            example_error(a).dt = max(result(valid1(:))./gdt(valid1(:)), gdt(valid1(:))./result(valid1(:)));
            
            example_error(a).sqr_raw = diff(valid2(:)).^2;
            example_error(a).rel_raw = diff(valid2(:))./gdt(valid2(:));
            example_error(a).dt_raw = max(result(valid2(:))./gdt(valid2(:)), gdt(valid2(:))./result(valid2(:)));
            
            example_error(a).sqr_ukn = diff(valid3(:)).^2;
            example_error(a).rel_ukn = diff(valid3(:))./gdt(valid3(:));
            example_error(a).dt_ukn = max(result(valid3(:))./gdt(valid3(:)), gdt(valid3(:))./result(valid3(:)));  
        end
        
        metric.rel = median(vertcat(example_error.rel));
        metric.rmse = sqrt(median(vertcat(example_error.sqr)));
        dt = vertcat(example_error.dt);
        metric.dt = [sum(dt<1.05)/length(dt) sum(dt<1.10)/length(dt) sum(dt<1.25)/length(dt) sum(dt<1.5625)/length(dt) sum(dt<1.9531)/length(dt)];
        
        metric.rel_raw = median(vertcat(example_error.rel_raw));
        metric.rmse_raw = sqrt(median(vertcat(example_error.sqr_raw)));
        dt = vertcat(example_error.dt_raw);
        metric.dt_raw = [sum(dt<1.05)/length(dt) sum(dt<1.10)/length(dt) sum(dt<1.25)/length(dt) sum(dt<1.5625)/length(dt) sum(dt<1.9531)/length(dt)];
        
        metric.rel_ukn = median(vertcat(example_error.rel_ukn));
        metric.rmse_ukn = sqrt(median(vertcat(example_error.sqr_ukn)));
        dt = vertcat(example_error.dt_ukn);
        metric.dt_ukn = [sum(dt<1.05)/length(dt) sum(dt<1.10)/length(dt) sum(dt<1.25)/length(dt) sum(dt<1.5625)/length(dt) sum(dt<1.9531)/length(dt)];
        
    case 'mp_render'
        load('matterporttestdata.mat');
        list = '../torch/data_list/mp_test_list_horizontal.txt';
        
        fp = fopen(list);
        tmp = textscan(fp, '%s');
        temp_list = tmp{1};
        fclose(fp);

        rs_list = cell(length(temp_list),1);
        for a = 1:length(temp_list)
            seg = regexp(temp_list{a}, '/', 'split');
            rs_list{a} = sprintf('%s/%s_%s_%s.png', result_folder, seg{end-2}, strrep(seg{end}(1:end-4), '_i', '_d'), suffix);  
        end
        
        example_error = repmat(struct('error',[]), length(temp_list), 1);
        for a = 1:length(temp_list)
            fprintf('%d/%d\n', a, length(temp_list));
            result = imread(rs_list{a});
            result = double(result)/4000;
            raw = testdata(a).raw;
            gdt = testdata(a).gdt;
            
            diff = abs(result-gdt);
            
            valid1 = gdt>0;
            valid2 = gdt>0 & raw>0;
            valid3 = gdt>0 & raw==0;
            
            example_error(a).sqr = diff(valid1(:)).^2;
            example_error(a).rel = diff(valid1(:))./gdt(valid1(:));
            example_error(a).dt = max(result(valid1(:))./gdt(valid1(:)), gdt(valid1(:))./result(valid1(:)));
            
            example_error(a).sqr_raw = diff(valid2(:)).^2;
            example_error(a).rel_raw = diff(valid2(:))./gdt(valid2(:));
            example_error(a).dt_raw = max(result(valid2(:))./gdt(valid2(:)), gdt(valid2(:))./result(valid2(:)));
            
            example_error(a).sqr_ukn = diff(valid3(:)).^2;
            example_error(a).rel_ukn = diff(valid3(:))./gdt(valid3(:));
            example_error(a).dt_ukn = max(result(valid3(:))./gdt(valid3(:)), gdt(valid3(:))./result(valid3(:)));  
            
            example_error(a).rel_ukn_med = median(example_error(a).rel_ukn);
            example_error(a).rmse_ukn_med = sqrt(median(example_error(a).sqr_ukn));
        end
        
        metric.rel = median(vertcat(example_error.rel));
        metric.rmse = sqrt(median(vertcat(example_error.sqr)));
        dt = vertcat(example_error.dt);
        metric.dt = [sum(dt<1.05)/length(dt) sum(dt<1.10)/length(dt) sum(dt<1.25)/length(dt) sum(dt<1.5625)/length(dt) sum(dt<1.9531)/length(dt)];
        
        metric.rel_raw = median(vertcat(example_error.rel_raw));
        metric.rmse_raw = sqrt(median(vertcat(example_error.sqr_raw)));
        dt = vertcat(example_error.dt_raw);
        metric.dt_raw = [sum(dt<1.05)/length(dt) sum(dt<1.10)/length(dt) sum(dt<1.25)/length(dt) sum(dt<1.5625)/length(dt) sum(dt<1.9531)/length(dt)];
        
        metric.rel_ukn = median(vertcat(example_error.rel_ukn));
        metric.rmse_ukn = sqrt(median(vertcat(example_error.sqr_ukn)));
        dt = vertcat(example_error.dt_ukn);
        metric.dt_ukn = [sum(dt<1.05)/length(dt) sum(dt<1.10)/length(dt) sum(dt<1.25)/length(dt) sum(dt<1.5625)/length(dt) sum(dt<1.9531)/length(dt)];
            
    otherwise
        fprintf('Unknown Dataset!\n')
end
end

