"""Check and fetch lastest model from a training machine.
python check_and_fetch_latest_model.py -gpus 0 1 2 3

Please add the SSH public key of this machine to the SSH authorized_keys file of the training machine.

Input:
remote_usr, remote_ip, remote_port, remote_work_dir, local_work_dir: for SSH.
gpus: for test.
wait_inter: waiting interval.
ckp_inter: checkpoint files interval.

Requirement:
see utils_ssh
"""
import os.path as osp
import argparse
import subprocess
import time

import utils_ssh


def check_latest_model(remote_usr, remote_ip, remote_dir, check_first, ckp_inter, wait_inter):
    """Return the latest model index.

    If the first model not found, wait for a while and check again automatically.
    """
    if_first_warning = True
    while True:
        first_model_path = osp.join(remote_dir, f'iter_{check_first}.pth')
        if utils_ssh.exists_remote(f'{remote_usr}@{remote_ip}', args.remote_port, first_model_path):  # the first model must exist, otherwise wait
            # check the real latest
            latest_idx = check_first
            latest_idx_check = check_first + ckp_inter
            while True:
                remote_model_path = osp.join(remote_dir, f'iter_{latest_idx_check}.pth')
                if utils_ssh.exists_remote(f'{remote_usr}@{remote_ip}', args.remote_port, remote_model_path):
                    latest_idx = latest_idx_check
                    latest_idx_check += ckp_inter
                else:
                    break
            break
        else:
            if if_first_warning:
                print(f'> the next first model not found: {check_first}; waiting...')
                if_first_warning = False
            time.sleep(wait_inter)
    return latest_idx


parser = argparse.ArgumentParser()
parser.add_argument('-remote_usr', type=str, default='root',)
parser.add_argument('-remote_ip', type=str, default='10.100.100.10',)
parser.add_argument('-remote_port', type=str, default='22',)
parser.add_argument('-remote_work_dir', type=str, default='<remote-work-dir>',)
parser.add_argument('-local_work_dir', type=str, default='<local-work-dir>',)

parser.add_argument('-gpus', metavar='N', type=int, nargs='+')
parser.add_argument('-wait_inter', type=int, default=15,)
parser.add_argument('-ckp_inter', type=int, default=1000,)
args = parser.parse_args()

gpus_str = ' '.join([str(gpu) for gpu in args.gpus])

if_first = True
while True:  # no ending check and test
    # check
    if if_first:
        check_first = args.ckp_inter
    latest_idx = check_latest_model(
        remote_usr=args.remote_usr,
        remote_ip=args.remote_ip,
        remote_dir=args.remote_work_dir,
        check_first=check_first,
        ckp_inter=args.ckp_inter,
        wait_inter=args.wait_inter)

    if if_first or (latest_idx != curr_ckp):
        print(f'> a latest model found: {latest_idx}')

        # scp
        remote_model_path = osp.join(args.remote_work_dir, f'iter_{latest_idx}.pth')
        local_model_path = osp.join(args.local_work_dir, f'iter_{latest_idx}.pth')
        cmd_ = f'scp -P {args.remote_port} {args.remote_usr}@{args.remote_ip}:{remote_model_path} {local_model_path}'
        pscp = subprocess.call(cmd_, shell=True)
        print(f'> scp done: {latest_idx}')

        curr_ckp = latest_idx

    if_test = if_first or not (ptest.poll() is None)
    if if_test:
        if if_first:
            print(f'> test the first model: {latest_idx}')
        else:
            print(f'> last test done. start to test: {latest_idx}')

        cmd_ = f'<test-command> -gpus {gpus_str}'
        ptest = subprocess.Popen(args=cmd_, shell=True)

        if_first = False
        check_first += args.ckp_inter
    else:
        time.sleep(args.wait_inter)
