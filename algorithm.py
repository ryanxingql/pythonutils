import utils

class BaseAlg():
    def __init__(self):
        super().__init__()
        
        self.rank = 0

    def create_model(self):
        pass

    def print_net(self):
        pass

    def create_loss_func(self, opts_dict):
        self.loss_lst = dict()
        for k_loss, loss_name in enumerate(opts_dict):
            loss_cls = getattr(utils, loss_name)
            loss_func = loss_cls(**opts_dict[loss_name]['opts'])
            loss_func = loss_func.to(self.rank)
            self.loss_lst[k_loss] = (opts_dict[loss_name]['weight'], loss_func)

    def create_optimizer(self):
        pass

    def save_model(self):
        pass

# ===
# Scheduler
# ===

# MultiStepRestartLR
# CosineAnnealingRestartLR
# https://github.com/RyanXingQL/SubjectiveQE-ESRGAN/blob/main/utils/deep_learning.py
