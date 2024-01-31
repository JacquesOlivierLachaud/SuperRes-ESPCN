from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.utils
import time
from torch.utils.tensorboard import SummaryWriter
from model import Net
from data import get_training_set, get_test_set
from helper import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train_path', type=str, default='', help='the path to the train dataset (if not specified, load/use BSDS300 dataset' )
parser.add_argument('--test_path', type=str, default='', help='the path to the train dataset (if not specified, load/use BSDS300 dataset' )
parser.add_argument('--model', type=str, default='', help='load a pre-trained model before going on on training' )
parser.add_argument('--run_name', type=str, default='', help='Gives a name to the run' )
opt = parser.parse_args()

print(opt)

torch.manual_seed(opt.seed)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"===> Using {device} device")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor, path=opt.train_path )
test_set  = get_test_set    (opt.upscale_factor, path=opt.test_path  )
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

run_name = 'runs/espcn-r'+str(opt.upscale_factor)+'-b'+str(opt.batchSize)
if opt.run_name != '':  run_name += '-'+opt.run_name
print('===> Preparing tensorboard: ' + run_name )
writer   = SummaryWriter(run_name)
# get some random training images
dataiter        = iter(training_data_loader)
inputs, targets = next(dataiter)
# create grid of images
input_grid      = torchvision.utils.make_grid(inputs [0:4])
target_grid     = torchvision.utils.make_grid(targets[0:4])
# show images
# matplotlib_imshow(input_grid, one_channel=False)
# matplotlib_imshow(target_grid, one_channel=False)
# write to tensorboard
writer.add_image('train_inputs',  input_grid)
writer.add_image('train_targets', target_grid)

if opt.model != '':
    print('===> Loading model '+opt.model)
    model = torch.load(opt.model).to( device )
else:
    print('===> Building model')
    model     = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
# SGD is much worse than Adam
# optimizer = optim.SGD(model.parameters(), lr=opt.lr)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=0.0001, verbose=True )

writer.add_graph( model, inputs.to( device ) )

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        # print( f'input={input.shape} target={target.shape}' )
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        if iteration % 4 == 3:    # every 4 batches...
                # ...log the running loss
                writer.add_scalar('training loss',
                                  loss.item() / 100,
                                  epoch * len(training_data_loader) + iteration )

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)

def test():
    avg_psnr = 0
    avg_psnr_bicubic = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            prediction = model(input)
            mse        = criterion(prediction, target)
            psnr       = 10 * log10(1 / mse.item())
            avg_psnr  += psnr
            bicubic    = v2.functional.resize( input.cpu(), size=( target.shape[-2], target.shape[-1] ), interpolation=v2.InterpolationMode.BICUBIC, antialias=True )
            mse_bicubic       = criterion(bicubic.to(device), target)
            psnr_bicubic      = 10 * log10(1 / mse_bicubic.item())
            avg_psnr_bicubic += psnr_bicubic
    avg_psnr /= len(testing_data_loader)
    avg_psnr_bicubic /= len(testing_data_loader)
    print("===> ESPCN   Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> bicubic Avg. PSNR: {:.4f} dB".format(avg_psnr_bicubic))
    return avg_psnr


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

t_start = time.time()
for epoch in range(1, opt.nEpochs + 1):
    t_epoch  = time.time()
    avg_loss = train(epoch)
    avg_psnr = test()
    checkpoint(epoch)
    scheduler.step( avg_loss )
    # scheduler.step() # MultiStepSR
    last_lr  = scheduler._last_lr
    t_now    = time.time()
    ellapsed_e = t_now - t_epoch
    ellapsed_b = t_now - t_start
    print( f"Time={ellapsed_b:.1f} s Time_epoch={ellapsed_e:.2f} s (current learning rate = {last_lr}" )
    writer.add_scalar('epoch average loss', avg_loss, epoch )
    writer.add_scalar('epoch average psnr', avg_psnr, epoch )
    writer.add_scalar('epoch lr', last_lr[0], epoch )

