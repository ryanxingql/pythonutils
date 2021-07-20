import sys
import time
import shutil
import random
import logging

import torch
import numpy as np


class CUDATimer:
    """
    https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
    """

    def __init__(self):
        self.inter_lst = None

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.reset_inter_lst()

    def reset_inter_lst(self):
        self.inter_lst = list()

    def start_record(self):
        self.start_event.record()

    def get_inter(self):
        self.end_event.record()
        torch.cuda.synchronize()
        et = self.start_event.elapsed_time(self.end_event) / 1000.  # the elapsed time (second) before end_event
        return et

    def record_inter(self):
        et = self.get_inter()
        self.inter_lst.append(et)

    def record_and_get_inter(self):
        et = self.get_inter()
        self.inter_lst.append(et)
        return et

    def get_ave_inter(self):
        return np.mean(self.inter_lst)

    def get_sum_inter(self):
        return np.sum(self.inter_lst)


def create_logger(log_path, rank=0, mode='a', fmt="%(asctime)s | %(message)s", datefmt="%Y/%m/%d %H:%M"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) if rank in [-1, 0] else logger.setLevel(logging.WARN)

    logFormatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # to log file
    fileHandler = logging.FileHandler(filename=log_path, mode=mode)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    # to screen
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger


def get_timestr():
    """Return current time str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


class Recorder:
    def __init__(self):
        self.result_lst = None
        self.reset()

    def reset(self):
        self.result_lst = []

    def record(self, result):
        self.result_lst.append(result)

    def get_ave(self):
        return np.mean(self.result_lst)


class Timer:
    def __init__(self):
        self.time_lst = None
        self.inter_lst = None

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


# Mkdir

def mkdir_archived(log_dir, if_del_arc=False):
    if log_dir.exists():  # if exists, rename the existing folder
        log_dir_pre = log_dir.parents[0]
        log_dir_name = log_dir.parts[-1]
        if if_del_arc:
            shutil.rmtree(log_dir)
            while True:  # rm may take time
                if not log_dir.exists():
                    break

        vs = 1
        log_dir_new = log_dir_pre / f'{log_dir_name}_v{vs}'
        while log_dir_new.exists():
            if if_del_arc:
                shutil.rmtree(log_dir_new)

            vs += 1
            log_dir_new = log_dir_pre / f'{log_dir_name}_v{vs}'

        if not if_del_arc:
            log_dir.rename(log_dir_new)
    log_dir.mkdir(parents=True)  # make log dir


# Random seed

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
