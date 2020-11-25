import math
import torch
import lpips
import numpy as np
from scipy import stats
from torch import nn as nn

class PCC():
    """Pearson correlation coefficient."""
    def __init__(self):
        self.help = (
            'Pearson correlation coefficient measures linear correlation '
            'between two variables X and Y. '
            'It has a value between +-1. '
            '+1: total positive linear correlation. '
            '0: no linear correlation. '
            '-1: total negative linear correlation. '
            'See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient'
            )
    
    def cal_pcc_two_imgs(self, x, y):
        """Calculate Pearson correlation coefficient of two images.
        
        Consider each pixel in x as a sample from a variable X, each pixel in y
        as a sample from a variable Y. Then an mxn image equals to mxn times
        sampling. 

        Input:
            x, y: two imgs (numpy array).
        Return:
            (cc value, p-value)

        Formula: https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.stats.pearsonr.html?highlight=pearson#scipy.stats.pearsonr

        Note: x/y should not be a constant! Else, the sigma will be zero, 
        and the cc value will be not defined (nan).
        """
        return stats.pearsonr(x.reshape((-1,)), y.reshape((-1,)))
    
    def _test(self):
        x = np.array([[3,4],[1,1]],dtype=np.float32)
        y = x + np.ones((2,2),dtype=np.float32)
        print(self.cal_pcc_two_imgs(x,y))

# ===
# Criterion
# ===

crit_lst = ['PSNR', 'LPIPS']

def return_crit_func(name, opts):
    assert (name in crit_lst), '> Not supported!'
    crit_func_cls = globals()[name]
    if opts is not None:
        return crit_func_cls(**opts)
    else:
        return crit_func_cls()

class PSNR(torch.nn.Module):
    """Input tensor. Return a float."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, x, y):
        mse = self.mse_func(x, y)
        psnr = 10 * math.log10(1 / mse.item())
        return psnr

class LPIPS(torch.nn.Module):
    """
    Args:
        spatial: return a score or a map of scores.
    """
    def __init__(self, net='alex', if_spatial=False, if_cuda=True):
        super().__init__()
        self.lpips_fn = lpips.LPIPS(net=net, spatial=if_spatial)
        if if_cuda:
            self.lpips_fn.cuda()

    def forward(self, x, y):
        return self.lpips_fn.forward(x, y).item()
    