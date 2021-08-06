import math
import random
from pathlib import Path
from functools import partial

import torch
import numpy as np
from cv2 import cv2
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .conversion import bgr2rgb


def _paired_random_crop(img_gts, img_lqs, h_patch, w_patch, if_center=False):
    """Apply the same cropping to GT and LQ image pairs.

    scale: cropped lq patch can be smaller than the cropped gt patch.

    List in, list out; ndarray in, ndarray out.
    """
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape

    assert (h_gt >= h_patch) and (w_gt >= w_patch), 'TARGET PATCH SIZE IS LARGER THAN THE IMAGE SIZE!'

    if if_center:
        top_idx = (h_lq - h_patch) // 2
        left_idx = (w_lq - w_patch) // 2
    else:  # randomly choose top and left coordinates for lq patch
        top_idx = random.randint(0, h_lq - h_patch)
        left_idx = random.randint(0, w_lq - w_patch)

    # crop
    img_lqs = [
        v[top_idx:top_idx + h_patch, left_idx:left_idx + w_patch, ...]
        for v in img_lqs
    ]
    img_gts = [
        v[top_idx:top_idx + h_patch, left_idx:left_idx + w_patch, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def _augment(img_lst, if_flip=True, if_rot=True):
    """Apply the same flipping and (or) rotation to all the imgs.
    
    Flipping is applied both x-axis and y-axis.
    Rotation can be 0, 90, 180 or 270 degrees.
    """

    def _main(img):
        if if_flip:
            cv2.flip(img, -1, img)  # in-place
        if if_rot:
            cv2.rotate(img, rot_code, img)
        return img

    if not isinstance(img_lst, list):
        img_lst = [img_lst]
    if_flip = if_flip and random.random() < 0.5
    if_rot = if_rot and random.random() < 0.5
    rot_code = random.choice([
        0,  # 90 degrees
        1,  # 180 degrees
        2,  # 270 degrees
    ]
    )
    img_lst = [_main(img) for img in img_lst]
    if len(img_lst) == 1:
        img_lst = img_lst[0]
    return img_lst


def _totensor(img_lst, if_bgr2rgb=True, if_float32=True):
    """(H W [BGR]) uint8 ndarray -> ([RGB] H W) float32 tensor
    
    List in, list out; ndarray in, ndarray out.
    """

    def _main(img):
        if if_bgr2rgb:
            img = bgr2rgb(img)
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if if_float32:
            img = img.float() / 255.
        return img

    if isinstance(img_lst, list):
        return [_main(img) for img in img_lst]
    else:
        return _main(img_lst)


class DistSampler(Sampler):
    """Distributed sampler that loads data from a subset of the dataset.

    Actually just generate idxs.
    Why enlarge? We only shuffle the dataloader before each epoch.
        Enlarging dataset can save the shuffling time.
    Support we have im00, im01, im02. We set ratio=3 and we have 2 workers.
        Enlarged ds: im00 01 02 00 01 02 00 01 02
        Worker 0: im00, im02, im01, im00, im02
        Worker 1: im01, im00, im02, im01, (im00)
    Enlargement is compatible with augmentation.
        Each sampling is different due to the random augmentation.
    Modified from torch.utils.data.distributed.DistributedSampler.

    Args:
        dataset size.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio.
    """

    def __init__(self, ds_size, num_replicas=None, rank=None, ratio=1):
        # do not & need not super the init of Sampler

        self.ds_size = ds_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # enlarged by ratio, and then divided by num_replicas
        self.num_samples = math.ceil(
            ds_size * ratio / self.num_replicas
        )
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        """
        For distributed training, shuffle the subset of each dataloader.
        For single-gpu training, no shuffling.
        """
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)  # shuffle based on self.epoch
        idxs = torch.randperm(self.total_size, generator=g).tolist()
        idxs = [idx % self.ds_size for idx in idxs]
        idxs = idxs[self.rank: self.total_size: self.num_replicas]
        return iter(idxs)

    def __len__(self):
        return self.num_samples  # for one rank


def create_dataloader(if_train, dataset, num_worker=None, batch_size=None, sampler=None, rank=None, seed=None):
    """Create dataloader.
    
    Dataloader is created for each rank.
        So num_worker and batch_size here are for one rank (one gpu).
    """
    if if_train:
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_worker,
            sampler=sampler,
            shuffle=False,  # sampler will shuffle at __iter__
            drop_last=True,
            pin_memory=True,  # must be True for prefetcher
        )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            _worker_init_fn,
            num_workers=num_worker,
            rank=rank,
            seed=seed
        )
    else:
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )
    return DataLoader(**dataloader_args)


def _worker_init_fn(worker_id, num_workers, rank, seed):
    """For reproducibility, fix seed of each worker.
    
    Seeds for different workers of all ranks are different.
    Suppose we have 2 ranks and 16 workers per rank.
        Rank 0:
            worker 0: seed + 16 * 0 + 0
            worker 1: seed + 16 * 0 + 1
            ...
            worker 15: seed + 16 * 0 + 15
        Rank 1:
            worker 0: seed + 16 * 1 + 0
            worker 1: seed + 16 * 1 + 1
            ...
            worker 15: seed + 16 * 1 + 15        
    """
    worker_seed = seed + num_workers * rank + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CPUPrefetcher:
    """CPU pre-fetcher."""

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def skip_front(self, niter, verbose=False):
        if verbose:
            pbar = tqdm(total=niter, ncols=80)
            for _ in range(niter):
                next(self.loader)
                pbar.update()
            pbar.close()
        else:
            for _ in range(niter):
                next(self.loader)

    def reset(self):
        self.loader = iter(self.ori_loader)


"""
class CUDAPrefetcher:
    '''CUDA pre-fetcher.

    Ref: https://github.com/NVIDIA/apex/issues/304#
    It may consume more GPU memory.
    
    Args:
        loader: Dataloader.
    '''
    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
"""


class DiskIODataset(Dataset):
    """Dataset using disk IO.
    
    gt_path and lq_path: relative paths to the dataset folder.
    opt_aug: True for training data, and False for test data.
    max_num: clip the dataset.
    if_train: if True, crop the images.
    """

    def __init__(self, gt_path, lq_path, if_train, max_num=-1, start_idx=0, aug=None, center_crop=None, padding=None):
        super().__init__()

        self.opts_aug = aug['opts'] if (aug is not None) else None
        self.opts_center_crop = center_crop['opts'] if ((center_crop is not None) and center_crop['if_crop']) else None
        self.opts_padding = padding['opts'] if ((padding is not None) and padding['if_pad']) else None

        # dataset path
        self.gt_path = Path(gt_path) if gt_path is not None else None
        self.lq_path = Path(lq_path)
        self.if_train = if_train

        # record data info
        self.data_info = dict(
            gt_path=[],
            lq_path=[],
            idx=[],
            name=[],
        )

        lq_lst = sorted(list(self.lq_path.glob('*.png')))
        if max_num != -1:
            lq_lst = lq_lst[start_idx:start_idx + max_num]  # default start_idx is 0
        else:
            lq_lst = lq_lst[start_idx:]
        self.im_num = len(lq_lst)

        for idx, lq_path in enumerate(lq_lst):
            name = lq_path.stem  # no .png
            gt_path = self.gt_path / (name + '.png') if self.gt_path is not None else None
            self.data_info['idx'].append(idx)
            self.data_info['gt_path'].append(gt_path)
            self.data_info['lq_path'].append(lq_path)
            self.data_info['name'].append(name)

    @staticmethod
    def _read_img(img_path):
        """Read im -> (H W [BGR]) uint8."""
        img_np = cv2.imread(str(img_path))
        return img_np

    def __getitem__(self, idx):
        gt_path = self.data_info['gt_path'][idx]
        img_gt = self._read_img(gt_path) if gt_path is not None else None  # (H W [BGR]) uint8
        img_lq = self._read_img(self.data_info['lq_path'][idx])

        # augmentation for training data
        # suppose that img_gt is not None
        if self.if_train:
            assert img_gt is not None, 'NO SUPERVISION!'
            img_gt, img_lq = _paired_random_crop(
                img_gt, img_lq, self.opts_aug['gt_h'], self.opts_aug['gt_w'], if_center=False,
            )
            img_lst = [img_lq, img_gt]  # gt is augmented jointly with lq
            img_lst = _augment(
                img_lst, self.opts_aug['if_flip'], self.opts_aug['if_rot'],
            )  # randomly crop
            img_lq, img_gt = img_lst[:]

        else:
            if self.opts_center_crop is not None:
                h, w, _ = img_gt.shape
                if h > self.opts_center_crop['h'] and w > self.opts_center_crop['w']:
                    if gt_path is not None:
                        img_gt, img_lq = _paired_random_crop(
                            img_gt, img_lq, self.opts_center_crop['h'], self.opts_center_crop['w'], if_center=True,
                        )
                    else:
                        img_gt, img_lq = _paired_random_crop(
                            img_lq, img_lq, self.opts_center_crop['h'], self.opts_center_crop['w'], if_center=True,
                        )

            if self.opts_padding is not None:
                h, w, _ = img_gt.shape
                h_ = int(np.ceil(h / self.opts_padding['mul']) * self.opts_padding['mul'])
                w_ = int(np.ceil(w / self.opts_padding['mul']) * self.opts_padding['mul'])
                dh = h_ - h if h_ > h else 0
                dw = w_ - w if w_ > w else 0
                # img_gt = np.pad(img_gt, ((h_ // 2, h - h_ // 2), (w_ // 2, w - w_ // 2)), mode='symmetric')
                img_lq = np.pad(img_lq, ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2), (0, 0)), mode='symmetric')

        # ndarray to tensor
        img_lst = [img_lq, img_gt] if img_gt is not None else [img_lq]
        img_lst = _totensor(img_lst)  # ([RGB] H W) float32)

        gt = img_lst[1] if img_gt is not None else -1
        return dict(
            lq=img_lst[0],
            gt=gt,
            name=self.data_info['name'][idx],  # dataloader will return it as a list (len is batch size)
            idx=self.data_info['idx'][idx],  # dataloader will return it as list-like tensor instead of numpy array
        )

    def __len__(self):
        return self.im_num


"""unfinished
class LMDBIODataset(Dataset):
    def __init__(self, opts_dict):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        dataroot = Path('data/DIV2K/')
        self.gt_root = dataroot / self.opts_dict['gt_path']
        self.lq_root = dataroot / self.opts_dict['lq_path']
        self.meta_info_path = self.gt_root / 'meta_info.txt'

        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

    @staticmethod
    def _read_img_bytes(img_bytes):
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_np = (img_np / 255.).astype(np.float32)
        return img_np

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )

        key = self.keys[index]
        img_bytes = self.file_client.get(key, 'gt')
        img_gt = self._read_img_bytes(img_bytes)  # (H W [BGR])
        img_bytes = self.file_client.get(key, 'lq')
        img_lq = self._read_img_bytes(img_bytes)  # (H W [BGR])

        # randomly crop
        gt_size = self.opts_dict['gt_sz']
        img_gt, img_lq = _paired_random_crop(
            img_gt, img_lq, gt_size,
            )

        # flip, rotate
        img_batch = [img_lq, img_gt] # gt joint augmentation with lq
        img_batch = _augment(
            img_batch, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_batch = _totensor(img_batch)  # ([RGB] H W)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,
            'gt': img_gt,
            }

    def __len__(self):
        return len(self.keys)
"""
