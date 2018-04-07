require 'image'
require 'utils'

function BatchIterator:nextBatchMatterport(set, config)

    local batch = {}
    batch.pr_color   = {}
    batch.cam_normal = {}
    batch.norm_valid = {}

    for i = 1, self.batch_size do
        -- get entry
        local entry = self:nextEntry(set)
        valid_data  = file_exists(entry.nx) and file_exists(entry.ny) and file_exists(entry.nz) and file_exists(entry.color)
        while not valid_data do
            entry = self:nextEntry(set)
            valid_data  = file_exists(entry.nx) and file_exists(entry.ny) and file_exists(entry.nz) and file_exists(entry.color)
        end 

        -- load data
        local pr_color = nil
        pr_color = image.load(entry.color)
        pr_color = image.scale(pr_color, 320, 256)
        pr_color = pr_color[{{1,3},{},{}}]
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end
        
        -- handle mix case
        if config.mix then
            raw_depth = image.load(entry.depth_raw)
            raw_depth = raw_depth:mul(65535/4000)
            step = 4 -- raw_depth:size(3)/320
            raw_depth = raw_depth:index(2,torch.range(1,raw_depth:size(2),step):long())
            raw_depth = raw_depth:index(3,torch.range(1,raw_depth:size(3),step):long())
            raw_valid = torch.eq(raw_depth,0):double()

            pr_color = torch.cat({pr_color, raw_depth, raw_valid}, 1) -- 5 channels: RGBDV
        end

        table.insert(batch.pr_color, pr_color)
        if config.verbose then
            print(string.format("pr_color max: %f, min: %f, size: %d %d", pr_color:max(), pr_color:min(), pr_color:size(2), pr_color:size(3)))
        end 

        if set == "train" then

            nx = image.load(entry.nx)
            ny = image.load(entry.ny)
            nz = image.load(entry.nz)
            local temp = torch.add(torch.pow(nx,2), torch.pow(ny,2))
            temp = torch.add(temp, torch.pow(nz,2))
            norm_valid = torch.gt(temp, 0.001)
            norm_valid = norm_valid:double()
            
            nx[torch.lt(norm_valid, 0.001)] = 0.5
            ny[torch.lt(norm_valid, 0.001)] = 0.5
            nz[torch.lt(norm_valid, 0.001)] = 0.5

            cam_normal = torch.Tensor(3, nx:size(2), nx:size(3))
            cam_normal[{1,{},{}}] = nx
            cam_normal[{2,{},{}}] = nz:mul(-1):add(1)
            cam_normal[{3,{},{}}] = ny

            cam_normal = cam_normal:add(-0.5):mul(2)
            
            step = cam_normal:size(3)/320
            cam_normal = cam_normal:index(2,torch.range(1,cam_normal:size(2),step):long())
            cam_normal = cam_normal:index(3,torch.range(1,cam_normal:size(3),step):long())
            norm_valid = norm_valid:index(2,torch.range(1,norm_valid:size(2),step):long())
            norm_valid = norm_valid:index(3,torch.range(1,norm_valid:size(3),step):long())     

            if config.mix then
                valid_output = torch.gt(norm_valid,0.5):long()
                valid_input  = torch.gt(raw_depth,0):long()

                valid_obs   = torch.cmul(valid_output, valid_input)
                valid_unobs = torch.cmul(valid_output, torch.eq(valid_input,0):long())

                if config.train_mask==2 then
                    norm_valid = valid_unobs:clone()
                elseif config.train_mask==3 then
                    norm_valid = valid_obs:clone()
                elseif config.train_mask==4 then
                    local bmask = torch.rand(raw_depth:size())
                    valid_random = torch.lt(bmask, config.random_ratio):long()
                    valid_obs = torch.cmul(valid_obs, valid_random)
                    norm_valid = valid_unobs + valid_obs
                end
            end

            table.insert(batch.cam_normal, cam_normal)
            table.insert(batch.norm_valid, norm_valid)
            if config.verbose then
                print(string.format("cam_normal max: %f, min: %f, size: %d %d", cam_normal:max(), cam_normal:min(), cam_normal:size(2), cam_normal:size(3)))
                print(string.format("norm_valid max: %f, min: %f, size: %d %d", norm_valid:max(), norm_valid:min(), norm_valid:size(2), norm_valid:size(3)))
                
            end 

        end
         
    end

    -- format img
    local ch, h, w
    ch, h, w= batch.pr_color[1]:size(1), batch.pr_color[1]:size(2), batch.pr_color[1]:size(3)
    batch.pr_color = torch.cat(batch.pr_color):view(self.batch_size, ch, h, w)

    if set == "train" then
        -- format camera normal
        ch, h, w = batch.cam_normal[1]:size(1), batch.cam_normal[1]:size(2), batch.cam_normal[1]:size(3)
        batch.cam_normal = torch.cat(batch.cam_normal):view(self.batch_size, ch, h, w)
        ch, h, w = batch.norm_valid[1]:size(1), batch.norm_valid[1]:size(2), batch.norm_valid[1]:size(3)
        batch.norm_valid = torch.cat(batch.norm_valid):view(self.batch_size, ch, h, w)
    end

    return batch
end
