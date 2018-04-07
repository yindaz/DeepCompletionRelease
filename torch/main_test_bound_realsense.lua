require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'
require 'image'

require 'BatchIterator'
require 'utils'
require 'utils_scannet'
require 'BatchIterator_scannet'
-- require 'hdf5'

local config = dofile('config_scannet.lua')
config = config.parse(arg)
print(config)
cutorch.setDevice(config.gpuid)

local tmp1 = split(config.test_model, "/")
config.result_path = config.result_path .. "/" .. string.sub(tmp1[#tmp1],1,-4)


config.result_path = config.result_path .. "_realsense_test_bound/"

os.execute("mkdir " .. config.result_path)

local model = dofile(config.model)(config) 

parameters, gradParameters = model:getParameters()
model.forwardnodes[24].data.module.modules[1]:__init(128,64,3,3,1,1,1,1)
model.forwardnodes[24].data.module.modules[2]:__init(64,nil,nil,nil)
model.forwardnodes[24].data.module.modules[4]:__init(64,3,3,3,1,1,1,1)
model.forwardnodes[24].data.module.modules[5]:__init(3,nil,nil,nil)

model:cuda()
parameters, gradParameters = model:getParameters()
parameters:copy(torch.load(config.test_model))

-- dataset
local train_data = {}
local test_data  = loadRealsense(config.test_file, config.root_path)
local batch_iterator = BatchIterator(config, train_data, test_data)
batch_iterator:setBatchSize(1)

local test_count = 0
local softmax_layer = nn.SoftMax():cuda()

while batch_iterator.epoch==0 and test_count<#batch_iterator.test.data do
    local batch = batch_iterator:nextBatchRealsense('test', config)
    local currName = batch_iterator:currentName('test')
    print(currName)
    local k = split(currName, "/")
    saveName = k[#k-1] .. "_" .. k[#k]
    print(string.format("Testing %s", saveName))
    

    local inputs = batch.pr_color
    inputs = inputs:contiguous():cuda()
    local outputs = model:forward(inputs)

    bound_est = outputs
    ch, h, w = bound_est:size(2), bound_est:size(3), bound_est:size(4)
    bound_est = bound_est:permute(1, 3, 4, 2):contiguous()
    bound_est = bound_est:view(-1, ch)

    bound_outputs = softmax_layer:forward(bound_est)
    bound_outputs = bound_outputs:view(1, h, w, ch)
    bound_outputs = bound_outputs:permute(1, 4, 2, 3):contiguous()
    bound_outputs = bound_outputs:view( ch, h, w)
    bound_outputs = bound_outputs:float()

    image.save(string.format("%s%s_bound_est.png", config.result_path, saveName), bound_outputs)

    test_count = test_count + 1
end

print("Finish!")






























