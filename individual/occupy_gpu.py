"""Occupy one gpu.
python main_gpu.py -ir 1 -g 0

Input:
gpu
upper_bound_ratio: maximal memory used ratio after occupation.
lower_bound_ratio: maximal memory used ratio to trigger occupation.
if_run: run computation and boost up the gpu utility.
gap_res: if not run, sleep for a while before the next check.


Requirement:
torch
see utils_gpu
"""
import time
import argparse

import torch
import torch.nn as nn

import utils_gpu


class NeuralNetwork(nn.Module):
    """只计算，不额外占用显存（无可学习参数及中间变量），避免溢出。"""

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            # nn.Conv2d(1,1,3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--upper_bound_ratio', '-ubr', type=float, default=0.9)  # 创建 tensor 后，总比例不能高于此经验上限，否则重新创建
parser.add_argument('--lower_bound_ratio', '-lbr', type=float, default=0.85)  # 如果低于下限，那么创建 tensor
# FAQ：可以将 ubr 和 lbr 合并吗？不能，容易导致反复横跳
parser.add_argument('--if_run', '-ir', type=int, default=0)  # 如果有 tensor，可以跑
parser.add_argument('--gap_res', '-gr', type=int, default=30)
args = parser.parse_args()

assert args.upper_bound_ratio >= args.lower_bound_ratio, 'ubr < lbr!'
assert args.if_run in [0, 1], 'wrong if_run!'

torch.cuda.set_device(args.gpu)

model = NeuralNetwork().to(args.gpu)
if_t_exist = False  # tensor 是否已创建
stat_pre = -1

while True:
    total, used, ratio_used = utils_gpu.check_gpu_memory(args.gpu)

    # 若显存不足下限，创建 tensor
    if ratio_used < args.lower_bound_ratio:
        if if_t_exist:  # 之前存在 tensor，之后会重建，因此 used 无效，要先删掉 tensor 再重算
            del tensor
            torch.cuda.empty_cache()
            total, used, ratio_used = utils_gpu.check_gpu_memory(args.gpu)
        t_volume = int(total * args.lower_bound_ratio - used) // 1024 // 1024  # Byte -> MiB

        tensor = torch.cuda.FloatTensor(t_volume, 1, 256, 1024)
        if_t_exist = True

        while True:  # 真实占用可能超过上限；为防止溢出，要根据实际情况调整 tensor 大小
            total, used, ratio_used = utils_gpu.check_gpu_memory(args.gpu)
            if ratio_used > args.upper_bound_ratio:
                del tensor
                torch.cuda.empty_cache()
                t_volume = int(t_volume * 0.999)  # 衰减
                tensor = torch.cuda.FloatTensor(t_volume, 1, 256, 1024)
            else:
                break

    # 输出信息
    timestr = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    _, _, ratio_used = utils_gpu.check_gpu_memory(args.gpu)
    ratioint = int(ratio_used * 100)
    if if_t_exist:
        if args.if_run == 1:
            msg = (f"{timestr} | gpu {args.gpu} | a {t_volume:5d} MiB tensor exists; running forever | ratio now: "
                   f"{ratioint:3d}%")
            stat_now = 0
        else:
            msg = (f"{timestr} | gpu {args.gpu} | a {t_volume:5d} MiB tensor exists; sleeping forever | ratio now: "
                   f"{ratioint:3d}%")
            stat_now = 1
    else:  # 从一开始，ratio 就高于 lbr，因此没有创建过 tensor
        if args.if_run == 1:
            msg = f"{timestr} | gpu {args.gpu} | nothing created; spying for running | ratio now: {ratioint:3d}%"
            stat_now = 2
        else:
            msg = f"{timestr} | gpu {args.gpu} | nothing created; spying for sleeping | ratio now: {ratioint:3d}%"
            stat_now = 3
    if stat_pre == -1:
        print(msg, end="", flush=True)
    else:
        if stat_now == stat_pre:
            print('\r' + msg, end="")
        else:
            print('\n' + msg, end="")
    stat_pre = stat_now

    # 跑或休眠
    if (args.if_run == 1) and if_t_exist:  # 有 tensor，且要跑
        model(tensor)
    else:
        time.sleep(args.gap_res)  # 既然不需要装 utils，那么可以休眠
