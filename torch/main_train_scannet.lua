require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'

require 'BatchIterator'
require 'utils'
require 'utils_scannet'
require 'BatchIterator_scannet'


-- config
local config = dofile('config_scannet.lua')
config = config.parse(arg)
cutorch.setDevice(config.gpuid)
print("Start: " .. config.ps)

-- model
local model_old = dofile(config.model)(config) 
parameters_old, gradParameters_old = model_old:getParameters()

if not isempty(config.finetune_model) then
    print('finetune from saved model weight...')
    parameters_old:copy(torch.load(config.finetune_model))
    print('set up learning rate...')
    config.optim_state.learningRate = config.finetune_init_lr
end

local model = dofile(config.model_nobn)(config)
parameters, gradParameters = model:getParameters()
parameters:copy(parameters_old[{{1,parameters:size(1)}}])

print(parameters:size(1))

-- modify model
model.forwardnodes[24].data.module.modules[1]:__init(128,64,3,3,1,1,1,1)
model.forwardnodes[24].data.module.modules[3]:__init(64,3,3,3,1,1,1,1)
model:cuda()
parameters, gradParameters = model:getParameters()


-- criterion
local criterion_n = nn.CosineEmbeddingCriterion():cuda()

-- dataset
if true then
    print('training with RENDERED ground truth surface normal...')
    train_data = loadScanNetRender(config.train_file, config.root_path)
    test_data  = loadScanNetRender(config.test_file, config.root_path)
end

local batch_iterator = BatchIterator(config, train_data, test_data)

-- logger
local logger = optim.Logger(config.log_path .. 'log', true)

-- main training
for it_batch = 1, math.floor(config.nb_epoch * #batch_iterator.train.data / config.batch_size) do

    local batch = batch_iterator:nextBatchScanNet('train', config)

    -- inputs and targets
    local inputs = batch.pr_color
    inputs = inputs:contiguous():cuda()
    
    local feval = function(x)
        -- prepare
        collectgarbage()
        if x ~= parameters then
            parameters:copy(x)
        end
        
        local est = model:forward(inputs)
        local valid = batch.norm_valid
        valid = valid:cuda()
        local gnd = batch.cam_normal
        gnd = gnd:cuda()

        bz, ch, h, w = est:size(1), est:size(2), est:size(3), est:size(4)
        est = est:permute(1,3,4,2):contiguous():view(-1,ch)
        local normalize_layer = nn.Normalize(2):cuda()
        est_n = normalize_layer:forward(est)
        gnd = gnd:permute(1,3,4,2):contiguous():view(-1,ch)

        f = criterion_n:forward({est_n, gnd}, torch.Tensor(est_n:size(1)):cuda():fill(1))
        df = criterion_n:backward({est_n, gnd}, torch.Tensor(est_n:size(1)):cuda():fill(1))
        df = df[1]
        df = normalize_layer:backward(est, df)

        valid = valid:view(-1,1):expandAs(df)
        df[torch.eq(valid,0)] = 0

        df = df:view(-1, h, w, ch)
        df = df:permute(1, 4, 2, 3):contiguous()

        gradParameters:zero()
        model:backward(inputs, df)

        -- print
        if it_batch % config.print_iters == 0 then
            print( it_batch, f)
        end

        -- log
        if it_batch % config.log_iters == 0 then
            logger:add{ f }
        end

        return f, gradParameters

    end

    -- optimizer
    optim.rmsprop(feval, parameters, config.optim_state)

    -- save
    if it_batch % config.snapshot_iters == 0 then
        print('saving model weight...')
        local filename
        filename = config.ps .. '_iter_' .. it_batch .. '.t7'
        torch.save(filename, parameters)
    end

    -- lr
    if it_batch % config.lr_decay == 0 then
        config.optim_state.learningRate = config.optim_state.learningRate / config.lr_decay_t
        config.optim_state.learningRate = math.max(config.optim_state.learningRate, config.optim_state.learningRateMin)
        print('decresing lr... new lr:', config.optim_state.learningRate)
    end
end

print('saving model weight...')
local filename
filename = config.ps .. 'final' .. '.t7'
torch.save(filename, parameters)

