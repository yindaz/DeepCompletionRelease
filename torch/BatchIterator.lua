require 'image'
require 'utils'

local BatchIterator = torch.class('BatchIterator')

function BatchIterator:__init(config, train_set, test_set)

    self.batch_size = config.batch_size or 128
    self.pixel_means = config.pixel_means or {0, 0, 0}
    self.mr = config.mr

    self.train = {}
    self.test  = {}

    self.train.data = train_set
    self.test.data  = test_set
    if #train_set > 0 then
        self.train.order = torch.randperm(#self.train.data)
    else
        self.train.order = torch.Tensor(0);
    end
    -- self.test.order  = torch.randperm(#self.test.data)
    self.test.order = torch.range(1,#self.test.data)
    self.train.id = 1
    self.test.id  = 1

    self.epoch = 0

end

function BatchIterator:setBatchSize(batch_size)
    self.batch_size = batch_size or 128
end

function BatchIterator:nextEntry(set)
    local i = self[set].i or 1
    self[set].i = i
    if i > #self[set].data then
        if set == "train" then
            self[set].order = torch.randperm(#self[set].data)
        end
        i = 1
        self.epoch = self.epoch + 1
    end

    local index = self[set].order[i]
    self[set].i = self[set].i + 1
    return self[set].data[index]
end

function BatchIterator:currentName(set)
    local i = self[set].i
    local index = self[set].order[i-1]
    return self[set].data[index].name
end

function BatchIterator:nextBatch(set, config)

    local batch = {}
    batch.input = {}
    batch.valid = {}
    batch.bound = {}

    for i = 1, self.batch_size do
        local entry = self:nextEntry(set)
        while not (file_exists(entry.input_file) and file_exists(entry.input_valid) and file_exists(entry.boundary)) do
            print(entry.input_file .. ' fail')
            entry = self:nextEntry(set)
        end

        if set == "train" then

            local valid = image.load(entry.input_valid)
            local boundraw = image.load(entry.boundary)

            -- define your data process here
            valid = valid:index(2,torch.range(1,valid:size(2),2):long())
            valid = valid:index(3,torch.range(1,valid:size(3),2):long())

            bound = boundraw:clone()
            bound = bound:mul(255):add(1)

            bound = bound:index(2,torch.range(1,bound:size(2),2):long())
            bound = bound:index(3,torch.range(1,bound:size(3),2):long())



            table.insert(batch.valid, valid)
            table.insert(batch.bound, bound)

            if config.verbose then
                print(string.format("valid max: %f, min: %f, size: %d %d", valid:max(), valid:min(), valid:size(2), valid:size(3)))
            end
        end

        local input = image.load(entry.input_file)

        -- process your input here
        input = input[{{1,3},{},{}}]
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                input[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                input[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end
        input = input:index(2,torch.range(1,input:size(2),2):long())
        input = input:index(3,torch.range(1,input:size(3),2):long())
        -- end

        table.insert(batch.input, input)
        if config.verbose then
            print(string.format("input max: %f, min: %f, size: %d %d", input:max(), input:min(), input:size(2), input:size(3)))
        end 
    end

    -- format img
    local ch, h, w = batch.input[1]:size(1), batch.input[1]:size(2), batch.input[1]:size(3)
    batch.input = torch.cat(batch.input, 1):view(self.batch_size, ch, h, w)

    if set == "train" then
        ch, h, w = batch.valid[1]:size(1), batch.valid[1]:size(2), batch.valid[1]:size(3)
        batch.valid  = torch.cat(batch.valid, 1):view(self.batch_size, ch, h, w)
        ch, h, w = batch.bound[1]:size(1), batch.bound[1]:size(2), batch.bound[1]:size(3)
        batch.bound  = torch.cat(batch.bound, 1):view(self.batch_size, ch, h, w)
    end

    return batch
end
