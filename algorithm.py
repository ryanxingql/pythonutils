import torch
import torch.nn as nn
from tqdm import tqdm
from cv2 import cv2
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP

from .conversion import tensor2im
from .system import print_n_log, Recoder
from .metrics import return_crit_func
from .deep_learning import return_optimizer, return_loss_func, return_scheduler

class BaseAlg():
    def __init__(self):
        #super().__init__()
        
        self.if_sched = self.opts_dict['train']['scheduler'].pop('if_sched') if self.if_train else False  # even if True, then if_train=False, it should be False

        if self.if_train:
            self.create_loss_func(
                opts_dict=self.opts_dict['train']['loss'],
                if_use_cuda=True,
                )
            
            params_lst = dict(
                net=self.model.module_lst['net'].parameters(),
                )
            self.create_optimizer(
                params_lst=params_lst,
                opts_dict=self.opts_dict['train']['optimizer'],
                )

            if self.if_sched:
                optim_lst = dict(
                    net=self.optim_lst['net'],
                    )
                self.create_scheduler(
                    optim_lst=optim_lst,
                    opts_dict=self.opts_dict['train']['scheduler'],
                    )

        if self.opts_dict['test']['criterion'] is not None:
            self.create_criterion(
                opts_dict=self.opts_dict['test']['criterion']
                )
        else:
            self.crit_lst = None

        if self.if_train:
            # 1/3: train from scratch
            if not self.opts_dict['train']['load_state']['if_load']:
                self.done_niter = 0
            # 2/3: train from ckp
            else:
                load_item_lst = ['module_net', 'optim_net',]
                
                if self.if_sched:
                    load_item_lst += ['sched_net',]

                self.done_niter = self.load_state(
                    ckp_load_path=self.opts_dict['train']['load_state']['ckp_load_path'],
                    load_item_lst=load_item_lst,
                    if_dist=True,
                    )
        # 3/3: test
        else:
            load_item_lst = ['module_net']
            self.done_niter = self.load_state(
                ckp_load_path=self.opts_dict['test']['ckp_load_path'],
                load_item_lst=load_item_lst,
                if_dist=True,
                )

    def add_graph(self, writer, data):
        self.set_eval_mode()
        writer.add_graph(self.model.module_lst['net'].module, data)

    def create_model(self, model_cls, if_train, opts_dict):
        """
        model_cls + opts_dict
        -> self.model
            self.model.module_lst
        -> DDP(module)
        """
        opts_dict_ = dict(
            if_train=if_train,
            opts_dict=opts_dict,
            )
        self.model = model_cls(**opts_dict_)

        for mod_item in self.model.module_lst:
            self.model.module_lst[mod_item].cuda()
            self.model.module_lst[mod_item] = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model.module_lst[mod_item]
                )  # convert all bn to syncbatchnorm
            self.model.module_lst[mod_item] = DDP(
                self.model.module_lst[mod_item],
                device_ids=[torch.cuda.current_device()],
                )

    def create_criterion(self, opts_dict):
        """
        opts_dict
        -> self.crit_lst

        Args:
            opts_dict (example):
                xxx
        """
        self.crit_lst = dict()
        for crit_item in opts_dict:
            fn = return_crit_func(crit_item, opts_dict[crit_item]['opts'])
            unit = opts_dict[crit_item]['unit']
            self.crit_lst[crit_item] = dict(
                fn=fn,
                unit=unit,
                )

    def create_loss_func(self, opts_dict, if_use_cuda=True):
        """
        opts_dict
        -> self.loss_lst
        -> (if_use_cuda) loss on cuda

        Args:
            opts_dict (example):
                xxx
        """
        self.loss_lst = dict()
        for loss_name in opts_dict:
            loss_func = return_loss_func(
                name=loss_name,
                opts=opts_dict[loss_name]['opts'],
                )
            if if_use_cuda:
                loss_func = loss_func.cuda()
            self.loss_lst[loss_name] = dict(
                weight=opts_dict[loss_name]['weight'],
                fn=loss_func,
                )

    def create_optimizer(self, params_lst, opts_dict):
        """
        opts_dict
        -> self.optim_lst
        -> (if TTUR) lr for gen / 2, lr for dis * 2.

        Args:
            params_lst (example):
                gen: params_g
                dis: params_d
            opts_dict (example):
                xxx
        """
        self.optim_lst = dict()
        for optim_item in opts_dict:         
            opts_dict_ = dict(
                name=opts_dict[optim_item]['type'],
                params=params_lst[optim_item],
                opts=opts_dict[optim_item]['opts'],
                )
            optim = return_optimizer(**opts_dict_)
            self.optim_lst[optim_item] = optim

    def create_scheduler(self, optim_lst, opts_dict):
        """
        opts_dict
        -> self.optim_lst

        Args:
            opts_dict (example):
                xxx
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

    def load_state(self, ckp_load_path, load_item_lst, if_dist=True):
        """
        ckp_load_path
        -> states

        states + load_item_lst
        -> load each item from states

        Args:
            load_item_lst (example):
                ['module_gen', 'module_dis', 'optim_gen', 'optim_dis', 'sched_gen'， 'sched_dis']
            if_dist: if load for distributed training.
        """
        states = torch.load(ckp_load_path)
        for load_item in load_item_lst:
            state_dict = states[load_item]

            # load module
            if 'module_' in load_item:
                item_name = load_item[7:]
                if ('module.' in list(state_dict.keys())[0]) and (not if_dist):  # multi-gpu pre-trained -> single-gpu training
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove module
                        new_state_dict[name] = v
                    self.model.module_lst[item_name].load_state_dict(new_state_dict)
                elif ('module.' not in list(state_dict.keys())[0]) and if_dist:  # single-gpu pre-trained -> multi-gpu training
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = 'module.' + k  # add module
                        new_state_dict[name] = v
                    self.model.module_lst[item_name].load_state_dict(new_state_dict)
                else:  # the same way of training
                    self.model.module_lst[item_name].load_state_dict(state_dict)
            
            # load optim
            elif 'optim_' in load_item:
                item_name = load_item[6:]
                self.optim_lst[item_name].load_state_dict(state_dict)
            
            # load sched
            elif 'sched_' in load_item:
                item_name = load_item[6:]
                self.sched_lst[item_name].load_state_dict(state_dict)
            
        return states['iter']

    def print_net(self, log_fp):
        if hasattr(self.model, 'msg_lst'):
            for msg_key in self.model.msg_lst:
                print_n_log(self.model.msg_lst[msg_key], log_fp)

    def save_state(self, ckp_save_path, iter, if_sched):
        """
        ckp_save_path
        -> save iter, modules, optims and scheds to ckp_save_path as states
        """
        states = dict(iter=iter)
        for mod_item in self.model.module_lst:
            states[f'module_{mod_item}'] = self.model.module_lst[mod_item].state_dict()
        for optim_item in self.optim_lst:
            states[f'optim_{optim_item}'] = self.optim_lst[optim_item].state_dict()
        if if_sched:
            for sched_item in self.sched_lst:
                states[f'sched_{sched_item}'] = self.sched_lst[sched_item].state_dict()
        torch.save(states, ckp_save_path)

    def set_eval_mode(self):
        """
        module.eval()
        """
        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].eval()

    def set_train_mode(self):
        """
        module.train()
        """
        for mod_key in self.model.module_lst:
            self.model.module_lst[mod_key].train()

    def test(
            self, test_fetcher, nsample_test, mod='normal', if_return_each=False, img_save_folder=None
            ):
        """
        baseline mod: test between src and dst.
        normal mod: test between src and tar.
        if_return_each: return result of each sample.

        note: temporally support bs=1, i.e., test one by one.
        """
        self.set_eval_mode()
        msg = ''
        write_dict_lst = []
        with torch.no_grad():
            flag_save_im = True

            if self.crit_lst is not None:
                for crit_name in self.crit_lst:
                    crit_fn = self.crit_lst[crit_name]['fn']
                    crit_unit = self.crit_lst[crit_name]['unit']

                    pbar = tqdm(total=nsample_test, ncols=80)
                    recorder = Recoder()
                    test_fetcher.reset()
                    
                    test_data = test_fetcher.next()
                    assert len(test_data['name']) == 1, 'Only support bs=1 for test!'
                    while test_data is not None:
                        im_gt = test_data['gt'].cuda(non_blocking=True)  # assume bs=1
                        im_lq = test_data['lq'].cuda(non_blocking=True)  # assume bs=1
                        im_name = test_data['name'][0]  # assume bs=1
                        
                        if mod == 'normal':
                            im_out = self.model.module_lst['net'](im_lq).clamp_(0., 1.)
                            perfm = crit_fn(torch.squeeze(im_out, 0), torch.squeeze(im_gt, 0))
                            
                            if flag_save_im and (img_save_folder is not None):  # save im
                                im = tensor2im(torch.squeeze(im_out, 0))
                                save_path = img_save_folder / (str(im_name) + '.png')
                                cv2.imwrite(str(save_path), im)
                        
                        elif mod == 'baseline':
                            perfm = crit_fn(torch.squeeze(im_lq, 0), torch.squeeze(im_gt, 0))
                        recorder.record(perfm)
                        
                        _msg = f'{im_name}: [{perfm:.3e}] {crit_unit:s}'
                        if if_return_each:
                            msg += _msg + '\n'
                        pbar.set_description(_msg)
                        pbar.update()
                        
                        test_data = test_fetcher.next()

                    flag_save_im = False
                    
                    # cal ave
                    ave_perfm = recorder.get_ave()
                    write_dict_lst.append(dict(
                        tag=f'{crit_name} (Test Set)',
                        scalar=ave_perfm,
                        ))
                    pbar.close()
                    if mod == 'normal':
                        msg += f'> {crit_name}: [{ave_perfm:.3e}] {crit_unit}\n'
                    elif mod == 'baseline':
                        msg += f'> baseline {crit_name}: [{ave_perfm:.3e}] {crit_unit}\n'
        
            else:  # only get tar
                pbar = tqdm(total=nsample_test, ncols=80)
                test_fetcher.reset()
                test_data = test_fetcher.next()
                assert len(test_data['name']) == 1, 'Only support bs=1 for test!'

                while test_data is not None:
                    im_lq = test_data['lq'].cuda(non_blocking=True)  # assume bs=1
                    im_name = test_data['name'][0]  # assume bs=1
                    im_out = self.model.module_lst['net'](im_lq).clamp_(0., 1.)
                    
                    if img_save_folder is not None:  # save im
                        im = tensor2im(torch.squeeze(im_out, 0))
                        save_path = img_save_folder / (str(im_name) + '.png')
                        cv2.imwrite(str(save_path), im)
                    
                    pbar.update()
                    test_data = test_fetcher.next()

                pbar.close()
                msg += f'> No ground-truth data; test done.\n'

        return msg.rstrip(), write_dict_lst

    def update_net_params(self, data, flag_step, inter_step):
        """available for simple loss func. for complex loss such as relativeganloss, please write your own func."""
        data_lq = data['lq'].cuda(non_blocking=True)
        data_gt = data['gt'].cuda(non_blocking=True)
        data_out = self.model.module_lst['net'](data_lq)

        self.gen_im_lst = dict(
            data_lq=data['lq'][:3],
            data_gt=data['gt'][:3],
            generated=data_out.detach()[:3].cpu().clamp_(0., 1.),
            )  # for torch.utils.tensorboard.writer.SummaryWriter.add_images: NCHW tensor is ok

        loss_total = 0
        for loss_item in self.loss_lst.keys():
            loss_dict = self.loss_lst[loss_item]
            opts_dict_ = dict(
                inp=data_out,
                ref=data_gt,
                )
            loss_unweighted = loss_dict['fn'](**opts_dict_)
            setattr(self, loss_item, loss_unweighted.item())  # for recorder
            loss_ = loss_dict['weight'] * loss_unweighted
            loss_total += loss_

        loss_total /= float(inter_step)  # multiple backwards and step once, thus mean
        loss_total.backward()  # must backward only once; otherwise the graph is freed after the first backward
        setattr(self, 'net_loss', loss_total.item())  # for recorder

        setattr(self, 'net_lr', self.optim_lst['net'].param_groups[0]['lr'])  # for recorder
        if flag_step:
            self.optim_lst['net'].step()
            self.optim_lst['net'].zero_grad()

    def update_lr(self):
        """Update lrs of all scheduler."""
        for sched_item in self.sched_lst:
            self.sched_lst[sched_item].step()

    def update_params(self, data, iter, flag_step, inter_step):
        self.net_loss = 0.  # for recorder
        self.net_lr = None
        self.gen_im_lst = dict()

        for param in self.model.module_lst['net'].parameters():
            param.requires_grad = True
        self.update_net_params(data=data, flag_step=flag_step, inter_step=inter_step)
        
        msg = (
            f'net_lr: [{self.net_lr:.3e}]; '
            f'net_loss: [{self.net_loss:.3e}]; '
            )
        write_dict_lst = [
            dict(tag='net_loss', scalar=self.net_loss),
            dict(tag='net_lr', scalar=self.net_lr),
            ]
        for loss_item in self.loss_lst:
            write_dict_lst.append(dict(
                tag=loss_item,
                scalar=getattr(self, loss_item),
                ))
            msg += f'{loss_item}: [{getattr(self, loss_item):.3e}]; '
        return msg[:-2], write_dict_lst, self.gen_im_lst
