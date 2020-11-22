import torch
import torch.distributed as dist
import torch.multiprocessing as tmp

# ===
# Multi-processing
# ===

def init_dist(local_rank=0, backend='nccl'):
    tmp.set_start_method('spawn', force=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)

# ===
# Optimizer
# ===

def return_optimizer(optim_type, params, optim_opts):
    assert (optim_type in ['Adam']), '> Not supported!'
    if optim_type == 'Adam':
        return torch.optim.Adam(params, **optim_opts)
