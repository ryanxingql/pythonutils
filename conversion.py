import torch
import numpy as np
import skimage.color as skc
from cv2 import cv2

# ===
# Image
# ===

def rgb2bgr(im):
    """RGB -> BGR.
    
    Input/Return: (..., C).
    """
    nc = im.shape[-1]
    assert (nc == 1 or nc == 3), 'Input format: (..., C)!'
    im = im[..., ::-1]
    return im

def bgr2rgb(im):
    """BGR -> RGB.
    
    Input/Return: (..., C).
    """
    nc = im.shape[-1]
    assert (nc == 1 or nc == 3), 'Input format: (..., C)!'
    im = im[..., ::-1]
    return im

"""OpenCV
def bgr2rgb(im):
    code = getattr(cv2, 'COLOR_BGR2RGB')
    im = cv2.cvtColor(im, code)
    return im

def rgb2bgr(im):
    code = getattr(cv2, 'COLOR_RGB2BGR')
    im = cv2.cvtColor(im, code)
    return im
"""

def tensor2im(t):
    """Tensor -> im.
    
    Input: tensor, (C H W), torch.float, mainly [0,1].
    Return: array, (H W C) for cv2.imwrite, np.uint8 ([0,255]).
    """
    im = t.cpu().detach().numpy()[::-1, :, :]
    im = im * 255.
    im = im.round()  # first round. directly astype will cut decimals
    im = im.clip(0, 255)  # else, -1 -> 255, -2 -> 254!
    im = im.astype(np.uint8)
    return im

# > im2tensor is usually conducted in dataset codes, and thus is omitted.

def rgb2ycbcr(rgb_im):
    """RGB -> YCbCr.

    Note:
        - Input/Return (..., 3) format.
        - Input/Return uint8 image, not [0,1] float array.
        - Input RGB, not BGR.
        - Returned YCbCr image has the same dimensions as that of the input RGB image, i.e., YUV 444P.
    
    Y is in the range [16,235]. Yb and Cr are in the range [16,240].

    This function produces the same results as Matlab's `rgb2ycbcr` function.
        It implements the ITU-R BT.601 conversion for standard-definition television.
        See: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
        In OpenCV, it implements a JPEG conversion.
        See: https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    """
    assert (rgb_im.dtype == np.uint8), 'Input uint8 image!'
    ycbcr_im = skc.rgb2ycbcr(rgb_im)
    return ycbcr_im

def bgr2ycbcr(bgr_im):
    rgb_im = bgr2rgb(bgr_im)
    ycbcr_im = rgb2ycbcr(rgb_im)
    return ycbcr_im

# ===
# Others
# ===

def dict2str(input_dict, indent=0):
    """Dict to string for printing options."""
    msg = ''
    indent_msg = ' ' * indent
    for k, v in input_dict.items():
        if isinstance(v, dict):  # still a dict
            msg += indent_msg + k + ':[\n'
            msg += dict2str(v, indent+2)
            msg += indent_msg + '  ]\n'
        else:  # the last level
            msg += indent_msg + k + ': ' + str(v) + '\n'
    return msg
