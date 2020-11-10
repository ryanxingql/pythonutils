import os
import time
import yaml
import argparse
import torch
import random
import numpy as np
from pathlib import Path

def get_timestr():
    """Return current time str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def arg2dict():
    """Receives args by argparse and YAML -> return dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option.yml', 
        help='Path to option YAML file.'
        )
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
    
    opts_dict['train']['rank'] = args.local_rank
    return opts_dict

def mkdir_archived(log_dir):
    if log_dir.exists():  # if exists, rename the existing folder
        log_dir_pre = log_dir.parents[0]
        log_dir_name = log_dir.parts[-1]
        log_dir_new = log_dir_pre / f'{log_dir_name}_v1'
        vs = 1
        while log_dir_new.exists():
            vs += 1
            log_dir_new = log_dir_pre / f'{log_dir_name}_v{vs}'
        log_dir.rename(log_dir_new) 
    log_dir.mkdir(parents=True)  # make log dir

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
