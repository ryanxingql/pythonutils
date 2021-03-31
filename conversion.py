import torch
import numpy as np
from cv2 import cv2

# ###
# image <-> numpy array / tensor
# ###

def imread(im_path):
    """Read image into numpy array.
    
    Return: (H W C) uint8 numpy array.
    """
    im = cv2.imread(im_path)

    im = im.astype(np.uint8)
    return im

def tensor2im(t):
    """Tensor -> im.
    
    Input: ([RGB] H W) torch.float tensor.
    Return: (H W [BGR]) np.uint8 array for cv2.imwrite.

    Note: RGB is default, since if_bgr2rgb of func:_totensor in dataset.py is True in default. 
    """
    t = t.cpu().detach()  # as copy in numpy

    im = t.numpy()[::-1, :, :]
    im = im.transpose(1, 2, 0)

    def _float2uint8(im):
        im *= 255.
        im = im.round()  # first round. directly astype will cut decimals
        im = im.clip(0, 255)  # else, -1 -> 255, -2 -> 254!
        im = im.astype(np.uint8)
        return im
    
    im = _float2uint8(im)
    return im

"""
im2tensor is usually conducted in dataset codes, and thus is omitted.
"""

# ###
# channel shuffling
# ###

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

# ###
# color space conversion
# ###

mat1 = np.array([
    [  65.481,  128.553,   24.966],
    [ -37.797,  -74.203,  112.   ],
    [ 112.   ,  -93.786,  -18.214],
    ]
) / 225.

mat2 = np.linalg.inv(mat1)

def rgb2ycbcr(im):
    """RGB -> YCbCr.

    Input: (H W C) uint8 image.
    
    Y is in the range [16,235]. Yb and Cr are in the range [16,240].
    See: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    """
    im = im.copy()
    im = im.astype(np.float64)
    
    im = im.dot(mat1.T)
    im[:,:,0] += 16.
    im[:,:,[1,2]] += 128.
    
    def _uint8(im):
        im = im.round()  # first round. directly astype will cut decimals
        im = im.clip(0, 255)  # else, -1 -> 255, -2 -> 254!
        im = im.astype(np.uint8)
        return im
    im = _uint8(im)
    return im

def bgr2ycbcr(im):
    """BGR -> YCbCr.
    
    Input: (H W C) uint8 image.
    """
    im = im.copy()
    im = im.astype(np.float64)
    
    im = bgr2rgb(im)
    im = rgb2ycbcr(im)

    im = im.astype(np.uint8)
    return im

def ycbcr2rgb(im):
    """YCbCr -> RGB. 444P.

    Input: (H W C) uint8 image.
    
    Y is in the range [16,235]. Yb and Cr are in the range [16,240].
    See: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    """
    im = im.copy()
    im = im.astype(np.float64)
    
    im[:,:,0] -= 16.
    im[:,:,[1,2]] -= 128.

    """
    mat = np.array([
        [255./219., 0., 255./224.],
        [255./219., -255./224.*1.772*0.114/0.587, -255./224.*1.402*0.299/0.587],
        [255./219., 255./224.*1.772, 0.],
        ])  # actually the inverse matrix of (that in rgb2ycbcr / 255.)
    """
    im = im.dot(mat2.T)  # error when using mat2 is smaller
    
    def _uint8(im):
        im = im.round()  # first round. directly astype will cut decimals
        im = im.clip(0, 255)  # else, -1 -> 255, -2 -> 254!
        im = im.astype(np.uint8)
        return im
    im = _uint8(im)
    return im

def ycbcr2bgr(im):
    """YCbCr -> BGR.
    
    Input: (H W C) uint8 image.
    """
    im = im.copy()
    im = im.astype(np.float64)

    im = ycbcr2rgb(im)
    im = rgb2bgr(im)

    im = im.astype(np.uint8)
    return im

def yuv420p2444p(y, u, v):
    """YUV 420P -> 444P.
    
    1 U/V serves for 4 Ys.

    Input: (H W 1) uint8 numpy array.
    """
    y, u, v = y.copy(), u.copy(), v.copy()
    assert y.shape[-1] == 1, 'Input (H W 1)!'

    u_up = np.zeros(y.shape, dtype=np.uint8)
    v_up = np.zeros(y.shape, dtype=np.uint8)
    uh, uw = u.shape[:2]
    for ih in range(uh):
        h_top = ih * 2
        for iw in range(uw):
            w_left = iw * 2

            u_value = u[ih,iw,0]
            v_value = v[ih,iw,0]

            u_up[h_top,w_left,0] = u_value
            u_up[h_top+1,w_left,0] = u_value
            u_up[h_top,w_left+1,0] = u_value
            u_up[h_top+1,w_left+1,0] = u_value
            
            v_up[h_top,w_left,0] = v_value
            v_up[h_top+1,w_left,0] = v_value
            v_up[h_top,w_left+1,0] = v_value
            v_up[h_top+1,w_left+1,0] = v_value
    yuv444p_im = np.concatenate((y, u_up ,v_up), axis=2)
    
    return yuv444p_im

yuv_type_list = ['420p', '444p']

def import_yuv(
        seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=True
    ):
    """Load Y, U, and V channels separately from a 8bit yuv420p video.
    
    Args:
        seq_path (str): .yuv (imgs) path.
        h (int): Height.
        w (int): Width.
        tot_frm (int): Total frames to be imported.
        yuv_type: 420p or 444p
        start_frm (int): The first frame to be imported. Default 0.
        only_y (bool): Only import Y channels.
    Return:
        y_seq, u_seq, v_seq (3 channels in 3 ndarrays): Y channels, U channels, 
        V channels.
    Note:
        YUV传统上是模拟信号格式, 而YCbCr才是数字信号格式.YUV格式通常实指YCbCr文件.
        参见: https://en.wikipedia.org/wiki/YUV
    """
    # setup params
    assert yuv_type in yuv_type_list, 'Not supported!'
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq

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
