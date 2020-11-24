import torch
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP

from .system import print_n_log
from .deep_learning import return_optimizer, return_loss_func, return_scheduler

class BaseAlg():
    def __init__(self):
        super().__init__()
        
        self.msg_lst = []

    def create_model(self, model_cls, if_train, opts_dict, rank):
        """
        Example Change:
            self.model.module_lst = {
                'gen': DDP wrapped generator,
                'dis': DDP wrapped discriminator,
                }
        """
        opts_dict_ = dict(
            if_train=if_train,
            opts_dict=opts_dict,
            )
        self.model = model_cls(**opts_dict_)

        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].to(rank)
            self.model.module_lst[mod_key] = DDP(self.model.module_lst[mod_key], device_ids=[rank])

    def load_model(self, load_lst, if_dist=True):
        """
        Args:
            load_lst (example):
                {
                    'gen': 'ckp_gen_10000.pt',
                    'dis': 'ckp_dis_10000.pt',
                }
            if_dist: if distributed training or not
        """
        for mod_key in load_lst:
            ckp_path = load_lst[mod_key]
            ckp = torch.load(ckp_path)
            state_dict = ckp['state_dict']
            if ('module.' in list(state_dict.keys())[0]) and if_dist:  # multi-gpu pre-trained -> single-gpu training
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove module
                    new_state_dict[name] = v
                self.model.module_lst[mod_key].load_state_dict(new_state_dict)
            elif ('module.' not in list(state_dict.keys())[0]) and if_dist:  # single-gpu pre-trained -> multi-gpu training
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k  # add module
                    new_state_dict[name] = v
                self.model.module_lst[mod_key].load_state_dict(new_state_dict)
            else:  # the same way of training
                self.model.module_lst[mod_key].load_state_dict(state_dict)

    def print_net(self, log_fp):
        for msg_key in self.model.msg_lst:
           print_n_log(self.model.msg_lst[msg_key], log_fp)

    def create_loss_func(self, opts_dict, if_use_cuda=True, rank=None):
        """
        Args:
            opts_dict (example):
                CharbonnierLoss:
                    weight: !!float 1e-2
                    opts:
                        eps: !!float 1e-6
                    
                VGGLoss:
                    weight: 1.
                    opts:
                        vgg_type: vgg19
                        layer_weights:
                            conv5_4: 1.
                        use_input_norm: True
                        perceptual_weight: 1.
                        style_weight: 0.
                    
                GANLoss:
                    weight: !!float 5e-3
                    opts:
                        real_label_val: 1.
                        fake_label_val: 0.
        """
        self.loss_lst = dict()
        for loss_name in opts_dict:
            loss_func = return_loss_func(name=loss_name, opts=opts_dict[loss_name]['opts'])
            if if_use_cuda:
                loss_func = loss_func.to(rank)
            self.loss_lst[loss_name] = dict(
                weight=opts_dict[loss_name]['weight'],
                func=loss_func,
                )

    def create_optimizer(self, params_lst, opts_dict):
        """
        Args:
            params_lst (example):
                {
                    'gen': params_g,
                    'dis': params_d,
                }
            opts_dict (example):
                TTUR:  # two time-scale update rule for GANs
                    if_ttur: True
                    lr: !!float 2e-4

                gen:
                    type: Adam
                    opts:
                    lr: ~
                        betas: [0.9, 0.999]

                dis:
                    type: Adam
                    opts:
                        lr: ~
                        betas: [0.9, 0.999]
        """
        opts_ttur = opts_dict.pop('TTUR')
        if_ttur = False
        if opts_ttur['if_ttur']:
            if_ttur = True
            lr = opts_ttur['lr']

        self.optim_lst = dict()
        for k_optim, optim_item in enumerate(opts_dict):
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
            self.optim_lst[optim_item] = optim

    def create_scheduler(self, optim_lst, opts_dict):
        """
        Args:
            opts_dict (example):
                if_sched: True

                gen:
                    type: MultiStepLR
                    opts:
                        milestones: [20,35,45,50]
                        gamma: 0.5

                dis:
                    type: MultiStepLR
                    opts:
                        milestones: [20,35,45,50]
                        gamma: 0.5
        """
        self.sched_lst = dict()
        for k_sched, sched_item in enumerate(opts_dict):
            opts_dict_ = dict(
                name=opts_dict[sched_item]['type'],
                optim=optim_lst[sched_item],
                opts=opts_dict[sched_item]['opts'],
                )
            sched = return_scheduler(**opts_dict_)
            self.sched_lst[sched_item] = sched

    def add_graph(self, writer, data):
        self.set_eval_mode()
        writer.add_graph(self.model.module_lst[list(self.model.module_lst.keys())[0]].module, data)

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
