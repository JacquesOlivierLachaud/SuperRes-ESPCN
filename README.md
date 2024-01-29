# SuperRes-ESPCN

A PyTorch implementation of single image super-resolution with a
sub-pixel convolutional neural network (ESPCN), based on CVPR 2016
paper [Real-Time Single Image and Video Super-Resolution Using an
Efficient Sub-Pixel Convolutional Neural
Network](https://arxiv.org/abs/1609.05158). A few details are also
inspired by this [ESPCN
implementation](https://github.com/leftthomas/ESPCN).

## Requirements

- tested with python 3.11.7
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c soumith
conda install pytorch torchvision cuda80 -c soumith # install it if you have installed cuda
```
- [TensorBoard](https://www.tensorflow.org/tensorboard)
```
conda install -y -c conda-forge tensorboard
```
- [OpenCV](https://opencv.org)
```
conda install -c conda-forge opencv
```

## Pre-trained models

Pre-trained models are in directories `models-up3`, `models-up4`,
depending on the sought upscale coefficient.

* `model-coco-l3-stanh.pth`

  This model is the original 3-layers, Tanh activation function, final
  sigmoid, and has been pre-trained on COCO dataset, with 57
  epochs. Final PSNR was 24.42 dB, compared to 23.8771 dB with bicubic
  interpolation.

## Usage

### Single image super-resolution

```
python sr-image.py
usage: sr-image.py [-h] --input_image INPUT_IMAGE --model MODEL
                   [--output_filename OUTPUT_FILENAME]
```

The upscale factor is determined by the given model.

## Datasets