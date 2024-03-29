from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Single image Super-Resolution with ESPCN')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, default='output.png', help='where to save the output image')
opt = parser.parse_args()

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"===> Using {device} device")

# Convert input image to YCbCr. Super-resolution is only done on Y channel.
img = Image.open(opt.input_image).convert('YCbCr')
y, cb, cr = img.split()

# Load model and convert input to tensor.
model = torch.load(opt.model)
img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
print( f"Input shape = {input.shape}" )

# Send model and input to device for computations.
model = model.to( device )
input = input.to( device )
out   = model(input).cpu()
print( f"Output shape = {out.shape}" )

# Compute final image by merging sr(Y) and bicubic interpolation of
# channels Cb and Cr.
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
print( f"Output image = {out_img_y.size}" )
out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

# Save image
out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
