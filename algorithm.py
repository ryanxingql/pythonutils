import torch
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP

from .system import print_n_log
from .deep_learning import return_optimizer, return_loss_func, return_scheduler

class BaseAlg():
    def __init__(self):
        super().__init__()

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

        for mod_item in self.model.module_lst:
            self.model.module_lst[mod_item].to(rank)
            self.model.module_lst[mod_item] = DDP(self.model.module_lst[mod_item], device_ids=[rank])

    def load_state(self, ckp_load_path, load_item_lst, if_dist=True):
        """
        Args:
            load_item_lst (example):
                ['module_gen', 'module_dis', 'optim_gen', 'optim_dis', 'sched_gen'， 'sched_dis']
            if_dist: if load for distributed training.
        """
        ckp = torch.load(ckp_load_path)
        for load_item in load_item_lst:
            state = ckp[load_item]
            # load module
            if 'module_' in load_item:
                item_name = load_item[7:]
                if ('module.' in list(state.keys())[0]) and if_dist:  # multi-gpu pre-trained -> single-gpu training
                    new_state = OrderedDict()
                    for k, v in state.items():
                        name = k[7:]  # remove module
                        new_state[name] = v
                    self.model.module_lst[item_name].load_state_dict(new_state)
                elif ('module.' not in list(state.keys())[0]) and if_dist:  # single-gpu pre-trained -> multi-gpu training
                    new_state = OrderedDict()
                    for k, v in state.items():
                        name = 'module.' + k  # add module
                        new_state[name] = v
                    self.model.module_lst[item_name].load_state_dict(new_state)
                else:  # the same way of training
                    self.model.module_lst[item_name].load_state_dict(state)
            # load optim
            elif 'optim_' in load_item:
                item_name = load_item[6:]
                self.optim_lst[item_name].load_state_dict(state)
            # load sched
            elif 'sched_' in load_item:
                item_name = load_item[6:]
                self.sched_lst[item_name].load_state_dict(state)

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
        for optim_item in opts_dict:
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
        for sched_item in opts_dict:
            opts_dict_ = dict(
                name=opts_dict[sched_item]['type'],
                optim=optim_lst[sched_item],
                opts=opts_dict[sched_item]['opts'],
                )
            sched = return_scheduler(**opts_dict_)
            self.sched_lst[sched_item] = sched

    def set_eval_mode(self):
        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].eval()

    def set_train_mode(self):
        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].train()

    def save_state(self, ckp_save_path, iter, if_sched):
        state = dict(iter=iter)
        for mod_item in self.model.module_lst:
            state[f'module_{mod_item}'] = self.model.module_lst[mod_item].state_dict()
        for optim_item in self.optim_lst:
            state[f'optim_{optim_item}'] = self.optim_lst[optim_item].state_dict()
        if if_sched:
            for sched_item in self.sched_lst:
                state[f'sched_{sched_item}'] = self.sched_lst[sched_item].state_dict()
        torch.save(state, ckp_save_path)

    def update_lr(self):
        for sched_item in self.sched_lst:
            self.sched_lst[sched_item].step()
