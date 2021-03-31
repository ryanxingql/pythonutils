import time
import yaml
import torch
import random
import argparse
import numpy as np

# ===
# Time & Recode
# ===

def get_timestr():
    """Return current time str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

class Timer():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.time_lst = [time.time()]
        self.inter_lst = []

    def record(self):
        self.time_lst.append(time.time())

    def record_inter(self):
        self.record()
        self.inter_lst.append(self.time_lst[-1] - self.time_lst[-2])

    def get_inter(self):
        return time.time() - self.time_lst[-1]

    def get_ave_inter(self):
        return np.mean(self.inter_lst)

    def get_total(self):
        return time.time() - self.time_lst[0]

class Recorder():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.result_lst = []

    def record(self, result):
        self.result_lst.append(result)
    
    def get_ave(self):
        return np.mean(self.result_lst)

# ===
# IO
# ===

def arg2dict():
    """Receives args by argparse and YAML -> return dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, default='option.yml', 
        help='Path to option YAML file.'
    )
    parser.add_argument(
        '--case', type=str, default='v1', 
        help='case for option.'
    )
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
    )
    args = parser.parse_args()
    
    with open(args.opt, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
        opts_dict = opts_dict[args.case]

    return opts_dict, args.local_rank

def print_n_log(msg, log_fp, if_new_line=True):
    """Display on screen and also log in file."""
    if if_new_line:
        msg += '\n'
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

# ===
# Dir
# ===

def mkdir_archived(log_dir):
    if log_dir.exists():  # if exists, rename the existing folder
        log_dir_pre = log_dir.parents[0]
        log_dir_name = log_dir.parts[-1]
        log_dir_new = log_dir_pre / f'{log_dir_name}-v1'
        vs = 1
        while log_dir_new.exists():
            vs += 1
            log_dir_new = log_dir_pre / f'{log_dir_name}-v{vs}'
        log_dir.rename(log_dir_new) 
    log_dir.mkdir(parents=True)  # make log dir

# ===
# Seed
# ===

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
