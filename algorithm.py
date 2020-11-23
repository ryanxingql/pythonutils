from torch.nn.parallel import DistributedDataParallel as DDP

from .system import print_n_log
from .deep_learning import return_optimizer, return_loss_func


class BaseAlg():
    def __init__(self):
        super().__init__()
        
        self.rank = 0
        self.msg_lst = []

    def create_model(self, model_cls, if_train, opts_dict, rank):
        opts_dict_ = dict(
            if_train=if_train,
            opts_dict=opts_dict,
            )
        self.model = model_cls(**opts_dict_)

        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].to(rank)
            self.model.module_lst[mod_key] = DDP(self.model.module_lst[mod_key], device_ids=[rank])

    def print_net(self, log_fp):
        for msg_key in self.msg_lst:
           print_n_log(self.msg_lst[msg_key], log_fp)

    def create_loss_func(self, opts_dict, if_use_cuda=True, rank=None):
        """
        Args:
            opts_dict:
                {
                    'CharbonnierLoss': {
                        'weight': 1e-2,
                        'opts': {
                            'eps': 1e-6,
                            },
                        },
                    'VGGLoss': {
                        'weight': 1.,
                        'opts': {
                            'vgg_type': 'vgg19',
                            'layer_weights': {
                                'conv5_4': 1.
                                },
                            'use_input_norm': True,
                            'perceptual_weight': 1.,
                            'style_weight': 0.,
                            },
                        },
                    'GANLoss': {
                        'weight': 5e-3,
                        'opts': {
                            'real_label_val: 1.,
                            'fake_label_val: 0.,
                            },
                        },
                }
        """
        self.loss_lst = dict()
        for k_loss, loss_name in enumerate(opts_dict):
            loss_func = return_loss_func(name=loss_name, opts=opts_dict[loss_name]['opts'])
            if if_use_cuda:
                loss_func = loss_func.to(rank)
            self.loss_lst[k_loss] = dict(
                name=loss_name,
                weight=opts_dict[loss_name]['weight'],
                func=loss_func,
                )

    def create_optimizer(self, params_lst, opts_dict):
        """
        Args:
            params_lst:
                {
                    'gen': params_g,
                    'dis': params_d,
                }
            opts_dict:
                {
                    'TTUR': {
                        'lr': 2e-4,
                        },
                    'gen': {
                        'type': 'Adam',
                        'opts': {
                            'lr': None,
                            'beta': [0.9, 0.999],
                            },
                        },
                    'dis': {
                        'type': 'Adam',
                        'opts': {
                            'lr': None,
                            'beta': [0.9, 0.999],
                            },
                        },
                }
                or:
                {
                    'gen': {
                        'type': 'Adam',
                        'opts': {
                            'lr': 2e-4,
                            'beta': [0.9, 0.999],
                            },
                        },
                    'dis': {
                        'type': 'Adam',
                        'opts': {
                            'lr': 2e-4,
                            'beta': [0.9, 0.999],
                            },
                        },
                }            
        """
        if_ttur = False
        if 'TTUR' in opts_dict.keys():
            if_ttur = True
            lr = opts_dict['TTUR']['lr']

        self.optim_lst = dict()
        k_optim = 0
        for optim_item in opts_dict:
            if optim_item in ['TTUR']:
                continue

            if if_ttur:
                if optim_item == 'gen':
                    new_lr = lr / 2.
                elif optim_item == 'dis':
                    new_lr = lr * 2.
                opts_dict[optim_item]['opts']['lr'] = new_lr
            
            opts_dict_ = dict(
                name=opts_dict[optim_item]['type'],
                params=params_lst[optim_item],
                opts=opts_dict[optim_item]['opts'],
                )
            optim = return_optimizer(**opts_dict_)
            self.optim_lst[k_optim] = dict(
                name=optim_item,
                optim=optim,
                )
            k_optim += 1

        self.cur_lr = lr

    def set_eval_mode(self):
        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].eval()

    def set_train_mode(self):
        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].train()

    def update_lr(self, epoch):
        pass

    def save_model(self):
        pass

# ===
# Scheduler
# ===

# MultiStepRestartLR
# CosineAnnealingRestartLR
# https://github.com/RyanXingQL/SubjectiveQE-ESRGAN/blob/main/utils/deep_learning.py
