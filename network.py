import torch
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def cal_num_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        return num_params

    #method: init_weights
