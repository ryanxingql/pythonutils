import math
import random
import torch
import numpy as np
from pathlib import Path
from cv2 import cv2
from functools import partial
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from .conversion import bgr2rgb

def _paired_random_crop(img_gts, img_lqs, gt_patch_size, scale=1):
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
    lq_patch_size = gt_patch_size // scale

    assert (h_gt == h_lq * scale) and (w_gt == w_lq * scale), 'Wrong scale!'
    assert (h_lq >= lq_patch_size) and (w_lq >= lq_patch_size), 'Target patch is larger than the input image!'

    # randomly choose top and left coordinates for lq patch
    top_lq = random.randint(0, h_lq - lq_patch_size)
    left_lq = random.randint(0, w_lq - lq_patch_size)
    top_gt, left_gt = int(top_lq * scale), int(left_lq * scale)

    # crop
    img_lqs = [
        v[top_lq:top_lq + lq_patch_size, left_lq:left_lq + lq_patch_size, ...]
        for v in img_lqs
        ]
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
        ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def _augment(imgs, if_flip=True, if_rot=True):
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

    if not isinstance(imgs, list):
        imgs = [imgs]
    if_flip = if_flip and random.random() < 0.5
    if_rot = if_rot and random.random() < 0.5
    rot_code = random.choice([
        0,  # 90 degrees
        1,  # 180 degrees
        2,  # 270 degrees
        ])
    imgs = [_main(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs

def _totensor(imgs, if_bgr2rgb=True, if_float32=True):
    """(H W [BGR]) uint8 ndarray -> ([RGB] H W) float32 tensor
    
    List in, list out; ndarray in, ndarray out.
    """
    def _main(img):
        if if_bgr2rgb:
            img = bgr2rgb(img)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if if_float32:
            img = img.float() / 255.
        return img

    if isinstance(imgs, list):
        return [_main(img) for img in imgs]
    else:
        return _main(imgs)

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
    def __init__(
            self, ds_size, num_replicas=None, rank=None, ratio=1
            ):
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

def create_dataloader(
        if_train,
        dataset,
        num_worker=None,
        batch_size=None,
        sampler=None,
        rank=None,
        seed=None,
        ):
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

class CPUPrefetcher():
    """CPU prefetcher."""
    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)

class CUDAPrefetcher():
    """CUDA prefetcher.

    Ref: https://github.com/NVIDIA/apex/issues/304#
    It may consums more GPU memory.
    
    Args:
        loader: Dataloader.
        opt (dict): Options.
    """
    def __init__(self, loader, rank):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.rank = rank
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
                    self.batch[k] = self.batch[k].to(
                        device=self.rank, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
    