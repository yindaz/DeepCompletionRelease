require 'image'
require 'utils'

function BatchIterator:nextBatchRealsense(set, config)

    local batch = {}
    batch.pr_color   = {}

    for i = 1, self.batch_size do
        -- get entry
        local entry = self:nextEntry(set)
        valid_data  = file_exists(entry.color)
        while not valid_data do
            print(entry.color)
            entry = self:nextEntry(set)
            valid_data  = file_exists(entry.color)
        end 

        -- load data
        local pr_color = nil
        pr_color = image.load(entry.color)
        pr_color = pr_color[{{1,3},{},{}}]
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end

        table.insert(batch.pr_color, pr_color)
        if config.verbose then
            print(string.format("pr_color max: %f, min: %f, size: %d %d", pr_color:max(), pr_color:min(), pr_color:size(2), pr_color:size(3)))
        end       
    end

    -- format img
    local ch, h, w
    ch, h, w= batch.pr_color[1]:size(1), batch.pr_color[1]:size(2), batch.pr_color[1]:size(3)
    batch.pr_color = torch.cat(batch.pr_color):view(self.batch_size, ch, h, w)

    return batch
end

function BatchIterator:nextBatchScanNet(set, config)

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

            table.insert(batch.cam_normal, cam_normal)
            table.insert(batch.norm_valid, norm_valid)
            if config.verbose then
                print(string.format("cam_normal max: %f, min: %f, size: %d %d", cam_normal:max(), cam_normal:min(), cam_normal:size(2), cam_normal:size(3)))
                print(string.format("norm_valid max: %f, min: %f, size: %d %d", norm_valid:max(), norm_valid:min(), norm_valid:size(2), norm_valid:size(3)))
                
            end 

        end


        -- load data
        local pr_color = nil
        pr_color = image.load(entry.color)
        pr_color = image.scale(pr_color, 320, 240)
        pr_color = pr_color[{{1,3},{},{}}]
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end

        table.insert(batch.pr_color, pr_color)
        if config.verbose then
            print(string.format("pr_color max: %f, min: %f, size: %d %d", pr_color:max(), pr_color:min(), pr_color:size(2), pr_color:size(3)))
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


function BatchIterator:nextBatchScanNetDepthDerivative(set, config)
    -- print(use_photo_realistic)
    -- local use_pr = use_photo_realistic or true
    -- print(use_photo_realistic)

    local batch = {}
    batch.pr_color   = {}
    batch.cam_normal = {}
    batch.norm_valid = {}
    batch.depth = {}
    batch.depth_valid = {}
    batch.derivative = {}

    for i = 1, self.batch_size do
        -- get entry
        local entry = self:nextEntry(set)
        valid_data  = file_exists(entry.depth) and file_exists(entry.color)
        while not valid_data do
            entry = self:nextEntry(set)
            valid_data  = file_exists(entry.depth) and file_exists(entry.color)
        end 

        if set == "train" then

            depth = image.load(entry.depth)
            -- depth_valid = torch.gt(depth, 0.001)
            -- depth_valid = depth_valid:double()
            
            depth = depth:mul(65535/4000)
            
            step = depth:size(3)/320
            -- print(string.format('step: %f', step))
            depth = depth:index(2,torch.range(1,depth:size(2),step):long())
            depth = depth:index(3,torch.range(1,depth:size(3),step):long())

            derivative, depth_valid = computeDerivative(depth)

            -- depth_valid = depth_valid:index(2,torch.range(1,depth_valid:size(2),step):long())
            -- depth_valid = depth_valid:index(3,torch.range(1,depth_valid:size(3),step):long())     
            derivative[torch.lt(depth_valid,0.01)] = 0


            table.insert(batch.depth, depth)
            table.insert(batch.derivative, derivative)
            table.insert(batch.depth_valid, depth_valid)
            if config.verbose then
                print(string.format("depth max: %f, min: %f, size: %d %d %d", depth:max(), depth:min(), depth:size(1), depth:size(2), depth:size(3)))
                print(string.format("derivative max: %f, min: %f, size: %d %d %d", derivative:max(), derivative:min(), derivative:size(1), derivative:size(2), derivative:size(3)))
                print(string.format("depth_valid max: %f, min: %f, size: %d %d %d", depth_valid:max(), depth_valid:min(), depth_valid:size(1), depth_valid:size(2), depth_valid:size(3)))              
            end 

        end


        -- load data
        local pr_color = nil
        pr_color = image.load(entry.color)
        pr_color = image.scale(pr_color, 320, 240)
        pr_color = pr_color[{{1,3},{},{}}]
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end
        -- pr_color = pr_color:index(2,torch.range(1,pr_color:size(2),4):long())
        -- pr_color = pr_color:index(3,torch.range(1,pr_color:size(3),4):long())


        table.insert(batch.pr_color, pr_color)
        if config.verbose then
            print(string.format("pr_color max: %f, min: %f, size: %d %d", pr_color:max(), pr_color:min(), pr_color:size(2), pr_color:size(3)))
        end       
    end

    -- format img
    local ch, h, w
    ch, h, w= batch.pr_color[1]:size(1), batch.pr_color[1]:size(2), batch.pr_color[1]:size(3)
    batch.pr_color = torch.cat(batch.pr_color):view(self.batch_size, ch, h, w)
    -- print(string.format("pr_color size: %d %d %d %d", self.batch_size, ch, h, w))

    if set == "train" then
        -- format camera normal
        ch, h, w = batch.depth[1]:size(1), batch.depth[1]:size(2), batch.depth[1]:size(3)
        batch.depth = torch.cat(batch.depth):view(self.batch_size, ch, h, w)
        -- print(string.format("norm_valid size: %d %d %d %d", self.batch_size, ch, h, w))
        ch, h, w = batch.depth_valid[1]:size(1), batch.depth_valid[1]:size(2), batch.depth_valid[1]:size(3)
        batch.depth_valid = torch.cat(batch.depth_valid):view(self.batch_size, ch, h, w)
        -- print(string.format("norm_valid size: %d %d %d %d", self.batch_size, ch, h, w))
        ch, h, w = batch.derivative[1]:size(1), batch.derivative[1]:size(2), batch.derivative[1]:size(3)
        batch.derivative = torch.cat(batch.derivative):view(self.batch_size, ch, h, w)
    end

    -- print(batch.norm_valid:size())
    return batch
end


function BatchIterator:nextBatchScanNetWNZ(set, config)

    local batch = {}
    batch.pr_color   = {}
    batch.wnz = {}
    batch.valid = {}

    for i = 1, self.batch_size do
        -- get entry
        local entry = self:nextEntry(set)
        valid_data  = file_exists(entry.wnz) and file_exists(entry.color) and file_exists(entry.depth)
        while not valid_data do
            entry = self:nextEntry(set)
            valid_data  = file_exists(entry.wnz) and file_exists(entry.color) and file_exists(entry.depth)
        end 

        if set == "train" then

            wnz = image.load(entry.wnz)
            depth = image.load(entry.depth)

            step = depth:size(3)/320
            depth = depth:index(2,torch.range(1,depth:size(2),step):long())
            depth = depth:index(3,torch.range(1,depth:size(3),step):long())

            step = wnz:size(3)/320
            wnz = wnz:index(2,torch.range(1,wnz:size(2),step):long())
            wnz = wnz:index(3,torch.range(1,wnz:size(3),step):long())

            wnz = wnz:add(-0.5):mul(2)       
            valid = torch.gt(depth, 0.0001)

            table.insert(batch.wnz, wnz)
            table.insert(batch.valid, valid)

            -- if config.verbose then
            --     print(string.format("depth max: %f, min: %f, size: %d %d %d", depth:max(), depth:min(), depth:size(1), depth:size(2), depth:size(3)))
            --     print(string.format("derivative max: %f, min: %f, size: %d %d %d", derivative:max(), derivative:min(), derivative:size(1), derivative:size(2), derivative:size(3)))
            --     print(string.format("depth_valid max: %f, min: %f, size: %d %d %d", depth_valid:max(), depth_valid:min(), depth_valid:size(1), depth_valid:size(2), depth_valid:size(3)))              
            -- end 

        end


        -- load data
        local pr_color = nil
        pr_color = image.load(entry.color)
        pr_color = image.scale(pr_color, 320, 240)
        pr_color = pr_color[{{1,3},{},{}}]
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end
        -- pr_color = pr_color:index(2,torch.range(1,pr_color:size(2),4):long())
        -- pr_color = pr_color:index(3,torch.range(1,pr_color:size(3),4):long())


        table.insert(batch.pr_color, pr_color)
        if config.verbose then
            print(string.format("pr_color max: %f, min: %f, size: %d %d", pr_color:max(), pr_color:min(), pr_color:size(2), pr_color:size(3)))
        end       
    end

    -- format img
    local ch, h, w
    ch, h, w= batch.pr_color[1]:size(1), batch.pr_color[1]:size(2), batch.pr_color[1]:size(3)
    batch.pr_color = torch.cat(batch.pr_color):view(self.batch_size, ch, h, w)
    -- print(string.format("pr_color size: %d %d %d %d", self.batch_size, ch, h, w))

    if set == "train" then
        -- format camera normal
        ch, h, w = batch.wnz[1]:size(1), batch.wnz[1]:size(2), batch.wnz[1]:size(3)
        batch.wnz = torch.cat(batch.wnz):view(self.batch_size, ch, h, w)
        -- print(string.format("norm_valid size: %d %d %d %d", self.batch_size, ch, h, w))
        ch, h, w = batch.valid[1]:size(1), batch.valid[1]:size(2), batch.valid[1]:size(3)
        batch.valid = torch.cat(batch.valid):view(self.batch_size, ch, h, w)
        -- print(string.format("norm_valid size: %d %d %d %d", self.batch_size, ch, h, w))
        -- ch, h, w = batch.derivative[1]:size(1), batch.derivative[1]:size(2), batch.derivative[1]:size(3)
        -- batch.derivative = torch.cat(batch.derivative):view(self.batch_size, ch, h, w)
    end

    -- print(batch.norm_valid:size())
    return batch
end

function BatchIterator:nextBatchScanNetNDOTV(set, config)

    local batch = {}
    batch.pr_color   = {}
    batch.ndotv = {}
    batch.valid = {}

    for i = 1, self.batch_size do
        -- get entry
        local entry = self:nextEntry(set)
        valid_data  = file_exists(entry.ndotv) and file_exists(entry.color)
        while not valid_data do
            entry = self:nextEntry(set)
            valid_data  = file_exists(entry.ndotv) and file_exists(entry.color)
        end 

        if set == "train" then

            ndotv = image.load(entry.ndotv)

            step = ndotv:size(3)/320
            ndotv = ndotv:index(2,torch.range(1,ndotv:size(2),step):long())
            ndotv = ndotv:index(3,torch.range(1,ndotv:size(3),step):long())

            ndotv = ndotv:add(-0.5):mul(2)       
            valid = torch.gt(ndotv, 0)

            table.insert(batch.ndotv, ndotv)
            table.insert(batch.valid, valid)

            -- if config.verbose then
            --     print(string.format("depth max: %f, min: %f, size: %d %d %d", depth:max(), depth:min(), depth:size(1), depth:size(2), depth:size(3)))
            --     print(string.format("derivative max: %f, min: %f, size: %d %d %d", derivative:max(), derivative:min(), derivative:size(1), derivative:size(2), derivative:size(3)))
            --     print(string.format("depth_valid max: %f, min: %f, size: %d %d %d", depth_valid:max(), depth_valid:min(), depth_valid:size(1), depth_valid:size(2), depth_valid:size(3)))              
            -- end 

        end


        -- load data
        local pr_color = nil
        pr_color = image.load(entry.color)
        pr_color = image.scale(pr_color, 320, 240)
        pr_color = pr_color[{{1,3},{},{}}]
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                pr_color[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end
        -- pr_color = pr_color:index(2,torch.range(1,pr_color:size(2),4):long())
        -- pr_color = pr_color:index(3,torch.range(1,pr_color:size(3),4):long())


        table.insert(batch.pr_color, pr_color)
        if config.verbose then
            print(string.format("pr_color max: %f, min: %f, size: %d %d", pr_color:max(), pr_color:min(), pr_color:size(2), pr_color:size(3)))
        end       
    end

    -- format img
    local ch, h, w
    ch, h, w= batch.pr_color[1]:size(1), batch.pr_color[1]:size(2), batch.pr_color[1]:size(3)
    batch.pr_color = torch.cat(batch.pr_color):view(self.batch_size, ch, h, w)
    -- print(string.format("pr_color size: %d %d %d %d", self.batch_size, ch, h, w))

    if set == "train" then
        -- format camera normal
        ch, h, w = batch.ndotv[1]:size(1), batch.ndotv[1]:size(2), batch.ndotv[1]:size(3)
        batch.ndotv = torch.cat(batch.ndotv):view(self.batch_size, ch, h, w)
        -- print(string.format("norm_valid size: %d %d %d %d", self.batch_size, ch, h, w))
        ch, h, w = batch.valid[1]:size(1), batch.valid[1]:size(2), batch.valid[1]:size(3)
        batch.valid = torch.cat(batch.valid):view(self.batch_size, ch, h, w)
        -- print(string.format("norm_valid size: %d %d %d %d", self.batch_size, ch, h, w))
        -- ch, h, w = batch.derivative[1]:size(1), batch.derivative[1]:size(2), batch.derivative[1]:size(3)
        -- batch.derivative = torch.cat(batch.derivative):view(self.batch_size, ch, h, w)
    end

    -- print(batch.norm_valid:size())
    return batch
end