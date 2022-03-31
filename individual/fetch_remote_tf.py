"""Fetch remote log and tensorboard files.
python fetch_remote_tf.py
tensorboard --logdir=./

Strongly recommend to collect all experiments into different subdirs.
"""
import os
import os.path as osp
import time
import datetime

time_step = 300

remote_usr_list = ['root',]
remote_ip_list = ['10.100.100.10',]
remote_port_list = ['2200',]
remote_log_dir_list = ['/remote/log/',]
remote_tf_dir_list = ['/remote/tf/',]
local_dir_list = ['/local/workspace/',]

for local_dir in local_dir_list:
    if not osp.exists(local_dir):
        os.mkdir(local_dir)

num_tf = len(remote_ip_list)
while True:
    for iter_tf in range(num_tf):
        usr = remote_usr_list[iter_tf]
        server = remote_ip_list[iter_tf]
        port = remote_port_list[iter_tf]
        tf_path = osp.join(remote_tf_dir_list[iter_tf], 'events.out.tfevents.*')
        log_path = osp.join(remote_log_dir_list[iter_tf], '*.log')
        local_dir = local_dir_list[iter_tf]
        
        cmd_ = f'scp -P {port} {usr}@{server}:{tf_path} {local_dir}'
        os.system(cmd_)
        cmd_ = f'scp -P {port} {usr}@{server}:{log_path} {local_dir}'
        os.system(cmd_)
        
    print(datetime.datetime.now())
    time.sleep(time_step)
