import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, opts_dict, if_train, infer_subnet='net'):
        super().__init__()

        self.opts_dict = opts_dict
        self.if_train = if_train
        self.infer_subnet = infer_subnet

    @staticmethod
    def _cal_num_params(module):
        num_params = 0
        for param in module.parameters():
            num_params += param.numel()
        return num_params

    def print_module(self, logger):
        for subnet in self.net:
            num_params = self._cal_num_params(self.net[subnet])
            logger.info(f'{subnet} is created with {num_params:d} params.')
