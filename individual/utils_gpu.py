"""
Requirement:
pynvml: pip install pynvml
"""
import pynvml
import time


pynvml.nvmlInit()


def check_gpu_memory(gpu):
    """Return gpu memory (Byte)."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used_ratio = mem_info.used / mem_info.total
    return mem_info.total, mem_info.used, mem_used_ratio


def check_empty_gpu(gpu_list=[0], min_mib=1024):
    """Return the first available gpu in the gpu list.
    If not found, return -1.

    Input:
    gpu_list: gpu list.
    min_mib: minimal MiB. If the gpu memory is below this number, then this gpu is available.
    """
    for gpu in gpu_list:
        _, mem_used, _ = check_gpu_memory(gpu)
        mem_used_mib = mem_used / 1024 / 1024
        if mem_used_mib < min_mib:
            return gpu
    return -1  # not found


def check_empty_gpu_and_wait(gpu_list=[0], check_inter=10, min_mib=1024):
    """Return an available gpu in the gpu list.
    If not found, wait for a while and check again automatically.

    Input:
    gpu_list: gpu list.
    check_inter: waiting interval.
    min_mib: minimal MiB. If the gpu memory is below this number, then this gpu is available.
    """
    if_first_warning = True
    while True:
        gpu = check_empty_gpu(gpu_list, min_mib)
        if gpu == -1:
            if if_first_warning:
                print(f'> sorry, no gpu is empty now! waiting...')
                if_first_warning = False
            time.sleep(check_inter)
        else:
            print(f'> gpu {gpu} is now empty; {check_inter} secs later we check again...')
            time.sleep(check_inter)  # sleep 10s

            gpu = check_empty_gpu([gpu], min_mib)  # check again
            if gpu == -1:
                print(f'> nope, this gpu is not empty!')
            else:
                return gpu
