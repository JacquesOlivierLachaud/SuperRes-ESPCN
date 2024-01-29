from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import cv2

def get_camera_res( cap ):
    """
    @return the current camera resolution 
    @param cap a cv2.VideoCapture object
    """
    return (cap.get(cv2.CAP_PROP_FRAME_WIDTH)),(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def roi( input, factor ):
    """
    @return the region of interest that is the zoomed region centered
    in the image input, as a numpy array
    
    @param input   the input image (a numpy array)
    @param factor  the zoom factor (>= 1.0)
    """
    if factor <= 1.0: return input
    h, w   = input.shape[ 0 ], input.shape[ 1 ]
    wp     = int( round( w / factor ) )
    hp     = int( round( h / factor ) )
    xp, yp = (w - wp) // 2, (h - hp) // 2
    return input[ yp:yp+hp, xp:xp+wp ]

def rescale( target, small ):
    """
    @return the image 'small' zoomed out to have the size of image 'target'
    @param small  the small image to zoom
    @param target the image whose size is targeted
    """
    h, w   = target.shape[ 0 ], target.shape[ 1 ]
    return cv2.resize( small, (w,h), interpolation=cv2.INTER_NEAREST )

def superres( model, device, cv_image ):
    """
    @return the super-resolution image of image 'cv_image', obtained with
    the given 'model' and 'device'.
    @param model the trained network that is used to compute the super-resolution
    @param device the device on which lies the model
    """
    rgb        = cv2.cvtColor( resized, cv2.COLOR_BGR2RGB)
    img_pil    = Image.fromarray( rgb ).convert('YCbCr')
    y, cb, cr  = img_pil.split()
    input      = img_to_tensor( y ).view(1, -1, y.size[1], y.size[0])
    input      = input.to( device )
    out        = model(input).cpu()
    out_img_y  = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y  = out_img_y.clip(0, 255)
    out_img_y  = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img    = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    np_image   = np.array(out_img)  
    output     = cv2.cvtColor( np_image, cv2.COLOR_RGB2BGR) 
    return output

def superres_doubled( model, device, cv_image ):
    """
    @return the doubled super-resolution image of image 'cv_image', obtained with
    the given 'model' and 'device', applied two times
    @param model the trained network that is used to compute the super-resolution
    @param device the device on which lies the model
    """
    rgb        = cv2.cvtColor( resized, cv2.COLOR_BGR2RGB)
    img_pil    = Image.fromarray( rgb ).convert('YCbCr')
    y, cb, cr  = img_pil.split()
    input      = img_to_tensor( y ).view(1, -1, y.size[1], y.size[0])
    input      = input.to( device )
    out        = model(input)
    out        = out.clip(0, 255)
    out        = model( out ).cpu()
    out_img_y  = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y  = out_img_y.clip(0, 255)
    out_img_y  = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img    = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    np_image   = np.array(out_img)  
    output     = cv2.cvtColor( np_image, cv2.COLOR_RGB2BGR)
    return output



# Training settings
parser = argparse.ArgumentParser(description='Super-Resolution with ESPCN video')
parser.add_argument('--model', type=str, required=True, help='model file to use')
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

print(f"===> Loading model {opt.model}")
model = torch.load(opt.model)
img_to_tensor = ToTensor()
model = model.to( device )
print( model )

# Start camera and change it to size 1280x720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
print( "===> camera resolution=", get_camera_res( cap ) )
if not cap.isOpened():
    print("Cannot open camera")
    exit()

upscale = 3
# nb= 0 : no superres, 1 : once sr, 2 : twice sr
nb      = 0
# mode= 0 : ESPCN, 1 : NEAREST, 2 : LINEAR, 3 : BICUBIC
mode    = 0 
zoom    = 1.0
font    = cv2.FONT_HERSHEY_SIMPLEX
    
while True:
    # Capture frame-by-frame
    ret, raw = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # zoom-in if necessary
    frame   = roi( raw, zoom )
    size    = frame.shape
    dsize   = ( size[1] // upscale, size[0] // upscale )
    ddsize  = ( dsize[0] // upscale, dsize[1] // upscale )
    tsize   = ( size[1], size[0] )
    if nb == 1:
        tsize = ( dsize[0] * upscale, dsize[1] * upscale )
    elif nb == 2:
        tsize = ( ddsize[0] * upscale * upscale, ddsize[1] * upscale * upscale )
    resized = frame
    if nb >= 1:
        # downsample once
        blur  = cv2.GaussianBlur( frame, (5,5), 0.245*upscale, 0.245*upscale, cv2.BORDER_DEFAULT)
        resized = cv2.resize( blur, dsize, interpolation = cv2.INTER_NEAREST)
    if nb == 2:
        # downsample twice
        blur  = cv2.GaussianBlur( resized, (5,5), 0.245*upscale, 0.245*upscale, cv2.BORDER_DEFAULT)
        resized = cv2.resize( blur, ddsize, interpolation = cv2.INTER_NEAREST)

    output = resized
    if mode == 0:
        if nb == 1:
            # compute super-resolution
            output = superres( model, device, resized )
        elif nb == 2:
            # compute super-resolution two-times
            output = superres_doubled( model, device, resized )
    elif mode == 1:
        output = cv2.resize( resized, tsize, interpolation=cv2.INTER_NEAREST )
    elif mode == 2:
        output = cv2.resize( resized, tsize, interpolation=cv2.INTER_LINEAR )
    elif mode == 3:
        output = cv2.resize( resized, tsize, interpolation=cv2.INTER_CUBIC )

    final = rescale( raw, output )
    str = "SR 1x1"
    if nb == 1:   str  = "SR 3x3"
    if nb == 2:   str  = "SR 9x9"
    if mode == 0: str += " ESPCN"
    if mode == 1: str += " NEAREST"
    if mode == 2: str += " BILINEAR"
    if mode == 3: str += " BICUBIC"
    cv2.putText( final, str, (10, final.shape[0]-20), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Display the resulting frame
    cv2.imshow('frame', final )
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):   break
    elif key == ord('0'): nb = 0
    elif key == ord('1'): nb = 1
    elif key == ord('2'): nb = 2
    elif key == ord('s'): mode = 0
    elif key == ord('n'): mode = 1
    elif key == ord('l'): mode = 2
    elif key == ord('c'): mode = 3        
    elif key == ord('+'): zoom *= 1.1
    elif key == ord('-'): zoom /= 1.1    
    if ( zoom < 1.0 ): zoom = 1.0
# while True:    
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

