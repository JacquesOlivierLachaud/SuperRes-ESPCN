from __future__ import print_function
import argparse
from PIL import Image
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Single image Super-Resolution with interpolation')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--interpolation', type=str, default='BICUBIC', help='interpolation to use, either NEAREST, BILINEAR, BICUBIC')
parser.add_argument('--upscale_factor', default=3, help='the chosen upscale factor')
parser.add_argument('--output_filename', type=str, default='output.png', help='where to save the output image')
opt = parser.parse_args()

# Convert input image to YCbCr. Super-resolution is made channel by channel
up        = opt.upscale_factor
img       = Image.open(opt.input_image).convert('YCbCr')
size      = img.size
tsize     = ( size[0]*up, size[1]*up )
print( f'{size} --> {tsize} with {opt.interpolation} function.' )
mode      = Image.BICUBIC
if   ( opt.interpolation == 'NEAREST' ): mode = Image.NEAREST
elif ( opt.interpolation == 'LINEAR' ):  mode = Image.BILINEAR
out_img   = img.resize( tsize, mode ).convert('RGB')

# Save image
out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
