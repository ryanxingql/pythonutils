import time
from pathlib import Path

import yaml
import matplotlib.pyplot as plt

with open('opt.yml', 'r') as fp:
    opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
    opts_dict = opts_dict['plot_curve_from_log']

log_fp = opts_dict['log_fp']
target_criteria = opts_dict['target_criteria']
save_dir = opts_dict['save_dir'] if opts_dict['save_dir'] is not None else './logs/'
current_dir = Path(__file__).resolve().parent
save_dir = (current_dir / save_dir).resolve()

log_fp = Path(log_fp).resolve()
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

log_name = log_fp.parent.name
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
save_fp = save_dir / f'valid_curve_{log_name}_{target_criteria}_{timestamp}.png'

skip_lines = 0
line_lst = log_fp.read_text().splitlines()[skip_lines:]

iter_lst = []
result_lst = []
for idx_line, line in enumerate(line_lst):
    if 'model is saved at' in line:
        model_path = Path(line[line.find('[') + 1: line.find(']')])
        pt_stem = model_path.stem
        iter = int(pt_stem.split('_')[-1])
        iter_lst.append(iter)

        next_line = line_lst[idx_line + 1]
        pos_ = next_line.find(target_criteria)
        result = float(next_line[next_line.find('[', pos_) + 1: next_line.find(']', pos_)])
        result_lst.append(result)

fig, ax = plt.subplots()
ax.plot(iter_lst, result_lst)
ax.set_title(log_name)
ax.set_xlabel('iter')
ax.set_ylabel(target_criteria)
ax.grid(axis='both')
plt.tight_layout()
fig.savefig(save_fp)
plt.show()
