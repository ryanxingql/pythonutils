from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from cv2 import cv2
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DisDP

from .conversion import tensor2im
from .system import Recorder, CUDATimer
from .metrics import return_crit_func
from .deep_learning import return_optimizer, return_loss_func, return_scheduler


class BaseAlg:
    """A basis class for deep-learning algorithms."""
    def __init__(self, opts_dict, model_cls, if_train, if_dist):
        super().__init__()

        self.opts_dict = opts_dict
        self.if_train = if_train
        self.if_dist = if_dist

        # Create model

        self.model = model_cls(opts_dict=opts_dict['network'], if_train=self.if_train)

        # Create loss functions, optimizers and schedulers

        if self.if_train:
            self.training_stage_lst = self.opts_dict['train']['niter']['name']
            self.loss_lst = dict()
            self.create_loss_func(opts_dict=self.opts_dict['train']['loss'], if_use_cuda=True)

            self.optim_lst = dict()
            self.create_optimizer(opts_dict=self.opts_dict['train']['optimizer'])

            self.if_sched = self.opts_dict['train']['scheduler'].pop('if_sched') if self.if_train else False  # if
            # if_train == False, it must be False
            if self.if_sched:
                self.sched_lst = dict()
                self.create_scheduler(opts_dict=self.opts_dict['train']['scheduler'])

            self.crit_lst = dict()
            self.create_criterion(opts_dict=self.opts_dict['val']['criterion'])

        else:
            self.crit_lst = dict()
            if self.opts_dict['test']['criterion'] is not None:
                self.create_criterion(opts_dict=self.opts_dict['test']['criterion'])

        # Load checkpoint

        if self.if_train:
            _if_load = False

            if self.opts_dict['train']['load_state']['if_load']:
                ckp_load_path = self.opts_dict['train']['load_state']['opts']['ckp_load_path']
                if ckp_load_path is None:
                    ckp_load_path = Path('exp') / self.opts_dict['exp_name'] / 'ckp_last.pt'

                if ckp_load_path.exists():
                    _if_load = True

            if not _if_load:  # no loading; train from scratch
                self.done_niter = 0
                self.best_val_perfrm = None

            else:  # load ckp
                if_load_net = True
                if_load_optim_ = self.opts_dict['train']['load_state']['opts']['if_load_optim']
                if_load_optim = True if if_load_optim_ else False
                if_load_sched = True if self.if_sched and if_load_optim_ else False
                self.done_niter, self.best_val_perfrm = self.load_state(ckp_load_path=ckp_load_path,
                                                                        if_load_net=if_load_net,
                                                                        if_load_optim=if_load_optim,
                                                                        if_load_sched=if_load_sched,
                                                                        if_dist=self.if_dist)

        else:  # test
            ckp_load_path = self.opts_dict['test']['ckp_load_path']
            if ckp_load_path is None:
                ckp_load_path = Path('exp') / self.opts_dict['exp_name'] / 'ckp_first_best.pt'

            if_load_net = True
            if_load_optim = False
            if_load_sched = False
            self.done_niter, _ = self.load_state(ckp_load_path=ckp_load_path, if_load_net=if_load_net,
                                                 if_load_optim=if_load_optim, if_load_sched=if_load_sched,
                                                 if_dist=self.if_dist)

        # move model to GPU
        # load ckp on cpu first, then move to gpu for saving memory: https://pytorch.org/docs/stable/generated/torch.load.html#torch.load
        self.model_cuda_ddp()

    def accum_gradient(self, module, stage, group, data, inter_step, **_):
        data_lq = data['lq'].cuda(non_blocking=True)
        data_gt = data['gt'].cuda(non_blocking=True)
        data_out = module(inp_t=data_lq, if_train=True)

        num_show = 3
        self._im_lst = dict(
            data_lq=data['lq'][:num_show],
            data_gt=data['gt'][:num_show],
            generated=data_out.detach()[:num_show].cpu().clamp_(0., 1.),
        )  # for torch.utils.tensorboard.writer.SummaryWriter.add_images: (N C H W) tensor is ok

        loss_total = torch.tensor(0., device="cuda")
        for loss_name in self.loss_lst[stage][group]:
            loss_dict = self.loss_lst[stage][group][loss_name]
            loss_unweighted = loss_dict['fn'](inp=data_out, ref=data_gt)

            setattr(self, f'{loss_name}_{group}', loss_unweighted.item())  # for recorder

            loss_ = loss_dict['weight'] * loss_unweighted
            loss_total += loss_

        loss_total /= float(inter_step)  # multiple backwards and step once, thus mean
        loss_total.backward()  # must backward only once; otherwise the graph is freed after the first backward
        setattr(self, f'loss_{group}', loss_total.item())  # for recorder

    def add_graph(self, writer, data):
        self.set_eval_mode()

        for subnet in self.model.net:
            if self.if_dist:
                writer.add_graph(self.model.net[subnet].module, data)
            else:
                writer.add_graph(self.model.net[subnet], data)

    def create_criterion(self, opts_dict):
        for crit_name in opts_dict:
            fn = return_crit_func(crit_name, opts_dict[crit_name]['opts'])
            unit = opts_dict[crit_name]['unit']
            if_focus = opts_dict[crit_name]['if_focus'] if 'if_focus' in opts_dict[crit_name] else False
            self.crit_lst[crit_name] = dict(fn=fn, unit=unit, if_focus=if_focus)

    def create_loss_func(self, opts_dict, if_use_cuda=True):
        for stage in self.training_stage_lst:
            if stage not in self.loss_lst:
                self.loss_lst[stage] = dict()

            for group in opts_dict[stage]:
                if group not in self.loss_lst[stage]:
                    self.loss_lst[stage][group] = dict()

                for loss_name in opts_dict[stage][group]:
                    opts_ = opts_dict[stage][group][loss_name]['opts'] if 'opts' in opts_dict[stage][group][loss_name] \
                        else dict()
                    loss_func_ = return_loss_func(name=loss_name, opts=opts_)
                    if if_use_cuda:
                        loss_func_ = loss_func_.cuda()

                    self.loss_lst[stage][group][loss_name] = dict(weight=opts_dict[stage][group][loss_name]['weight'],
                                                                  fn=loss_func_)

    def create_optimizer(self, opts_dict):
        assert len(self.model.net) == 1, 'RE-WRITE THIS FUNC TO SUPPORT MULTI SUB-NETS!'

        for stage in self.training_stage_lst:
            if stage not in self.optim_lst:
                self.optim_lst[stage] = dict()

            for group in opts_dict[stage]:
                if group not in self.optim_lst[stage]:
                    self.optim_lst[stage][group] = dict()

                opts_dict_ = dict(
                    name=opts_dict[stage][group]['name'],
                    params=self.model.net[self.model.infer_subnet].parameters(),  # all parameters
                    opts=opts_dict[stage][group]['opts'],
                )
                optim_ = return_optimizer(**opts_dict_)
                self.optim_lst[stage][group] = optim_  # one optimizer for one group in one stage

    def create_scheduler(self, opts_dict):
        for stage in self.training_stage_lst:
            self.sched_lst[stage] = dict() if stage not in self.sched_lst else self.sched_lst[stage]

            for group in opts_dict[stage]:
                self.sched_lst[stage][group] = dict() if group not in self.sched_lst[stage] \
                    else self.sched_lst[stage][group]

                opts_dict_ = dict(
                    name=opts_dict[stage][group]['name'],
                    optim=self.optim_lst[stage][group],
                    opts=opts_dict[stage][group]['opts'],
                )
                sched_ = return_scheduler(**opts_dict_)
                self.sched_lst[stage][group] = sched_

    def load_state(self, ckp_load_path, if_load_net=True, if_load_optim=False, if_load_sched=False, if_dist=True):
        states = torch.load(ckp_load_path, map_location='cpu')  # load on cpu to save memory. https://pytorch.org/docs/stable/generated/torch.load.html#torch.load

        if if_load_net:  # load network
            for subnet in self.model.net:
                state_dict = states['network'][subnet]

                if 'module.' in list(state_dict.keys())[0]:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove module
                        new_state_dict[name] = v

                else:
                    new_state_dict = state_dict

                """
                if ('module.' in list(state_dict.keys())[0]) and \
                        (not if_dist):  # multi-gpu pre-trained -> single-gpu training
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove module
                        new_state_dict[name] = v

                elif ('module.' not in list(state_dict.keys())[0]) and \
                        if_dist:  # single-gpu pre-trained -> multi-gpu training
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = 'module.' + k  # add module
                        new_state_dict[name] = v

                else:  # the same way of training
                    new_state_dict = state_dict
                """

                self.model.net[subnet].load_state_dict(new_state_dict)

        if if_load_optim:
            for stage in self.training_stage_lst:
                for group in self.optim_lst[stage]:
                    state_dict = states['optim'][stage][group]
                    self.optim_lst[stage][group].load_state_dict(state_dict)

                    for state in self.optim_lst[stage][group].state.values():  # although model is loaded on cpu, the optim and sched should be loaded on gpu
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        if if_load_sched:
            for stage in self.training_stage_lst:
                for group in self.sched_lst[stage]:
                    state_dict = states['sched'][stage][group]
                    self.sched_lst[stage][group].load_state_dict(state_dict)

                    for state in self.sched_lst[stage][group].state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        idx_iter = states['idx_iter']
        best_val_perfrm = states['best_val_perfrm']

        del states
        torch.cuda.empty_cache()

        print(f'ckp [{ckp_load_path}] loaded.')
        return idx_iter, best_val_perfrm

    def model_cuda_ddp(self):
        """Move to GPU; DDP if needed."""
        for subnet in self.model.net:
            self.model.net[subnet].cuda()
            if self.if_dist:
                self.model.net[subnet] = nn.SyncBatchNorm.convert_sync_batchnorm(self.model.net[subnet])
                # convert all bn to sync-batch-norm before wrapping network with DDP
                self.model.net[subnet] = DisDP(self.model.net[subnet], device_ids=[torch.cuda.current_device()])

    def save_state(self, ckp_save_path, best_val_perfrm, idx_iter, if_sched):
        states = dict(idx_iter=idx_iter, best_val_perfrm=best_val_perfrm, network=None, optim=dict(), sched=dict())

        states['network'] = dict()
        for subnet in self.model.net:
            states['network'][subnet] = self.model.net[subnet].state_dict()

        for stage in self.training_stage_lst:
            if stage not in states['optim']:
                states['optim'][stage] = dict()
            if stage not in states['sched']:
                states['sched'][stage] = dict()

            for group in self.optim_lst[stage]:
                states['optim'][stage][group] = self.optim_lst[stage][group].state_dict()

                if if_sched:
                    states['sched'][stage][group] = self.sched_lst[stage][group].state_dict()

        torch.save(states, ckp_save_path)

    def set_eval_mode(self):
        for subnet in self.model.net:
            self.model.net[subnet].eval()

    def set_train_mode(self):
        for subnet in self.model.net:
            self.model.net[subnet].train()

    @torch.no_grad()
    def test(self, data_fetcher, num_samples, if_baseline=False, if_return_each=False, img_save_folder=None,
             if_train=True):
        """
        test baseline
            True: test dst with ref=src.
            False: test tar with ref=src.

        if_return_each: return result of each sample.

        note: temporally support bs=1, i.e., test one by one.
        """
        if if_baseline or if_train:
            assert self.crit_lst is not None, 'NO METRICS!'

        if self.crit_lst is not None:
            if_tar_only = False
            msg = 'dst vs. src | ' if if_baseline else 'tar vs. src | '
        else:
            if_tar_only = True
            msg = 'only get dst | '

        report_dict = None

        recorder_dict = dict()
        for crit_name in self.crit_lst:
            recorder_dict[crit_name] = Recorder()

        write_dict_lst = []
        timer = CUDATimer()

        self.set_eval_mode()

        data_fetcher.reset()
        test_data = data_fetcher.next()
        assert len(test_data['name']) == 1, 'ONLY SUPPORT bs==1!'

        pbar = tqdm(total=num_samples, ncols=100)

        while test_data is not None:
            im_lq = test_data['lq'].cuda(non_blocking=True)  # assume bs=1
            im_name = test_data['name'][0]  # assume bs=1

            timer.start_record()
            if if_tar_only:
                im_out = self.model.net[self.model.infer_subnet](inp_t=im_lq, if_train=False).clamp_(0., 1.)
                timer.record_inter()
            else:
                im_gt = test_data['gt'].cuda(non_blocking=True)  # assume bs=1
                if if_baseline:
                    im_out = im_lq
                else:
                    im_out = self.model.net[self.model.infer_subnet](inp_t=im_lq, if_train=False).clamp_(0., 1.)
                timer.record_inter()

                _msg = f'{im_name} | '

                for crit_name in self.crit_lst:
                    crit_fn = self.crit_lst[crit_name]['fn']
                    crit_unit = self.crit_lst[crit_name]['unit']

                    perfm = crit_fn(torch.squeeze(im_out, 0), torch.squeeze(im_gt, 0))
                    recorder_dict[crit_name].record(perfm)

                    _msg += f'[{perfm:.3e}] {crit_unit:s} | '

                _msg = _msg[:-3]
                if if_return_each:
                    msg += _msg + '\n'
                pbar.set_description(_msg)

            if img_save_folder is not None:  # save im
                im = tensor2im(torch.squeeze(im_out, 0))
                save_path = img_save_folder / (str(im_name) + '.png')
                cv2.imwrite(str(save_path), im)

            pbar.update()
            test_data = data_fetcher.next()
        pbar.close()

        if not if_tar_only:
            for crit_name in self.crit_lst:
                crit_unit = self.crit_lst[crit_name]['unit']
                crit_if_focus = self.crit_lst[crit_name]['if_focus']

                ave_perfm = recorder_dict[crit_name].get_ave()
                msg += f'{crit_name} | [{ave_perfm:.3e}] {crit_unit} | '

                write_dict_lst.append(dict(tag=f'{crit_name} (val)', scalar=ave_perfm))

                if crit_if_focus:
                    report_dict = dict(ave_perfm=ave_perfm, lsb=self.crit_lst[crit_name]['fn'].lsb)

        ave_fps = 1. / timer.get_ave_inter()
        msg += f'ave. fps | [{ave_fps:.1f}]'
        if if_train:
            assert report_dict is not None
            return msg.rstrip(), write_dict_lst, report_dict
        else:
            return msg.rstrip()

    def update_params(self, stage, data, if_step, inter_step, additional):
        msg = ''
        tb_write_dict_lst = []
        for group in self.optim_lst[stage]:
            # Accumulate gradients

            self.accum_gradient(module=self.model.net[self.model.infer_subnet], stage=stage, group=group, data=data,
                                inter_step=inter_step, additional=additional)

            item_ = getattr(self, f'loss_{group}')
            msg += f'{group} loss: [{item_:.3e}] | '
            tb_write_dict_lst.append(dict(tag=f'loss_{group}', scalar=item_))

            for loss_name in self.loss_lst[stage][group]:
                item_ = getattr(self, f'{loss_name}_{group}')
                msg += f"{loss_name}_{group}: [{item_:.3e}] | "
                tb_write_dict_lst.append(dict(tag=f'{loss_name}_{group}', scalar=item_))

            # Update params

            if if_step:
                self.optim_lst[stage][group].step()

                item_ = self.optim_lst[stage][group].param_groups[0]['lr']  # for recorder
                msg += f"lr_{group}: [{item_:.3e}] | "
                tb_write_dict_lst.append(dict(tag=f'lr_{group}', scalar=item_))

                self.optim_lst[stage][group].zero_grad()  # empty the gradients for this group

            # Update learning rate (scheduler)

            if self.if_sched:
                self.sched_lst[stage][group].step()

        return msg[:-3], tb_write_dict_lst, self._im_lst
