require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'

require 'BatchIterator'
require 'utils'

-- config
local config = dofile('config.lua')
-- print(arg)
config = config.parse(arg)
-- print(config)
cutorch.setDevice(config.gpuid)
print("Start: " .. config.ps)

-- model
local model = dofile(config.model)(config) 
model:cuda()
parameters, gradParameters = model:getParameters()

print("Intializing with pre-trained model")
parameters:copy(torch.load(config.pretrain_file))

print("Modify pretrained model")
model.forwardnodes[24].data.module.modules[1]:__init(128,64,3,3,1,1,1,1)
model.forwardnodes[24].data.module.modules[2]:__init(64,nil,nil,nil)
model.forwardnodes[24].data.module.modules[4]:__init(64,3,3,3,1,1,1,1)
model.forwardnodes[24].data.module.modules[5]:__init(3,nil,nil,nil)

model:cuda()
parameters, gradParameters = model:getParameters()

-- resume training
if config.resume_training then
    print('loading saved model weight...')
    parameters:copy(torch.load(config.saved_model_weights))
    config.optim_state = torch.load(config.saved_optim_state)
end

if config.finetune then
    print('finetune from saved model weight...')
    parameters:copy(torch.load(config.finetune_model))
    print('set up learning rate...')
    config.optim_state.learningRate = config.finetune_init_lr
end

-- criterion
local criterion_b = nn.CrossEntropyCriterion(torch.Tensor({1,100,100})):cuda()

-- dataset
local train_data = loadData(config.train_file, config)
local test_data  = loadData(config.test_file, config)
local batch_iterator = BatchIterator(config, train_data, test_data)

-- logger
local logger = optim.Logger(config.log_path .. 'log', true)

-- main training
for it_batch = 1, math.floor(config.nb_epoch * #batch_iterator.train.data / config.batch_size) do

    local batch = batch_iterator:nextBatch('train', config)

    -- inputs and targets
    local inputs = batch.input
    inputs = inputs:contiguous():cuda()
    
    local feval = function(x)
        -- prepare
        collectgarbage()
        if x ~= parameters then
            parameters:copy(x)
        end
        
        -- forward propagation
        local est = model:forward(inputs)
        local valid = batch.valid
        valid = valid:cuda()
        local gnd = batch.bound
        gnd = gnd:cuda()

        bz, ch, h, w = est:size(1), est:size(2), est:size(3), est:size(4)
        est = est:permute(1,3,4,2):contiguous():view(-1,ch)
        gnd = gnd:permute(1,3,4,2):contiguous():view(-1,1)

        f  = criterion_b:forward(est,gnd)
        df = criterion_b:backward(est,gnd)

        valid = valid:view(-1,1):expandAs(df)
        df[torch.eq(valid,0)] = 0

        df = df:view(-1, h, w, ch)
        df = df:permute(1,4,2,3):contiguous()

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
        filename = config.ps .. 'iter_' .. it_batch .. '.t7'
        torch.save(filename, parameters)
        filename = config.ps .. 'iter_' .. it_batch .. '_state.t7'
        torch.save(filename, config.optim_state)
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
filename = config.ps .. 'final' .. '_state.t7'
torch.save(filename, config.optim_state)
