import torch
import torch.distributed as dist
import torch.multiprocessing as tmp

# ===
# Multi-processing
# ===

def init_dist(local_rank=0, backend='nccl'):
    tmp.set_start_method('spawn')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)

# ===
# Scheduler
# ===

# MultiStepRestartLR
# CosineAnnealingRestartLR
# https://github.com/RyanXingQL/SubjectiveQE-ESRGAN/blob/main/utils/deep_learning.py