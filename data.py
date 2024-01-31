from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
import torch
from torchvision.transforms import v2
from dataset import DatasetFromFolder


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return v2.Compose([
        v2.CenterCrop(crop_size),
        v2.GaussianBlur( upscale_factor*2+1, sigma=0.245*float(upscale_factor) ),
        v2.Resize(crop_size // upscale_factor, interpolation=v2.InterpolationMode.NEAREST),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        # ToTensor(),
    ])


def target_transform(crop_size):
    return v2.Compose([
        v2.CenterCrop(crop_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        # ToTensor(),
    ])


def get_training_set(upscale_factor, path ):
    if ( path == '' ):
        root_dir  = download_bsd300() 
        train_dir = join(root_dir, "train")
    else:
        train_dir = path
    print( f'===> train_path={train_dir}' )
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor, path ):
    if ( path == '' ):
        root_dir = download_bsd300() 
        test_dir = join(root_dir, "test")
    else:
        test_dir = path
    print( f'===> test_path={test_dir}' )
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
