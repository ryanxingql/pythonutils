import random
import torch
import numpy as np
from pathlib import Path
from torch.utils import data as data 
from cv2 import cv2
from .conversion import bgr2rgb

def _paired_random_crop(img_gts, img_lqs, gt_patch_size, scale=1):
    """Apply the same cropping to GT and LQ image pairs.

    scale: cropped lq patch can be smaller than the cropped gt patch.

    List in, list out; ndarray in, ndarray out.

    v0.0
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
    
    v0.0
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
    print(if_flip, if_rot)
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

    v0.0
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

def _read_img(img_path):
    """Read im -> (H W [BGR]) uint8

    v0.0
    """
    img_np = cv2.imread(str(img_path))
    return img_np

def _read_img_bytes(img_bytes):
    """
    unfinished
    """
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_np = (img_np / 255.).astype(np.float32)
    return img_np

class DiskIODataset(data.Dataset):
    """Dataset using disk IO.
    gt_path and lq_path: relative paths to the dataset folder.
    opt_aug: True for training data, and False for test data.

    v0.0
    """
    def __init__(self, gt_path, lq_path, aug):
        super().__init__()

        # dataset path
        self.gt_path = Path(gt_path)
        self.lq_path = Path(lq_path)

        # record data info
        self.data_info = {
            'gt_path': [],
            'lq_path': [],
            'idx': [],
            'name': [],
            }

        gt_lst = sorted(list(self.gt_path.glob('*.png')))
        self.gt_num = len(gt_lst)
        
        for idx, gt_path in enumerate(gt_lst):
            name = gt_path.stem  # no .png
            lq_path = self.lq_path / (name + '.png')
            self.data_info['idx'].append(idx)
            self.data_info['gt_path'].append(gt_path)
            self.data_info['lq_path'].append(lq_path)
            self.data_info['name'].append(name)

        self.aug = aug

    def __getitem__(self, idx):
        img_gt = _read_img(self.data_info['gt_path'][idx])  # (H W [BGR]) uint8
        img_lq = _read_img(self.data_info['lq_path'][idx])

        # augmentation for training data
        if self.aug['if_aug']:
            img_gt, img_lq = _paired_random_crop(
                img_gt, img_lq, self.aug['gt_size'],
                )
            img_batch = [img_lq, img_gt] # gt is augmented jointly with lq
            img_batch = _augment(
                img_batch, self.aug['if_flip'], self.aug['if_rot']
                )

        # ndarray to tensor
        img_batch = [img_lq, img_gt]
        img_batch = _totensor(img_batch)  # ([RGB] H W) float32

        return {
            'lq': img_batch[0],
            'gt': img_batch[1],
            'name': self.data_info['name'][idx], 
            'idx': self.data_info['idx'][idx], 
            }

    def __len__(self):
        return self.gt_num

class LMDBIODataset(data.Dataset):
    """
    unfinished
    """
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

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )

        key = self.keys[index]
        img_bytes = self.file_client.get(key, 'gt')
        img_gt = _read_img_bytes(img_bytes)  # (H W [BGR])
        img_bytes = self.file_client.get(key, 'lq')
        img_lq = _read_img_bytes(img_bytes)  # (H W [BGR])

        # randomly crop
        gt_size = self.opts_dict['gt_size']
        img_gt, img_lq = paired_random_crop(
            img_gt, img_lq, gt_size,
            )

        # flip, rotate
        img_batch = [img_lq, img_gt] # gt joint augmentation with lq
        img_batch = augment(
            img_batch, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_batch = totensor(img_batch)  # ([RGB] H W)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,
            'gt': img_gt,
            }

    def __len__(self):
        return len(self.keys)
    