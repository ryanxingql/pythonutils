from pathlib import Path

import matplotlib.pyplot as plt


log_fp = Path('/home/x/data/pycharm/PowerQE/exp/arcnn_div2k_qp37/log_train.log').resolve()
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
        result = float(next_line[next_line.find('[') + 1: next_line.find(']')])
        result_lst.append(result)

fig, ax = plt.subplots()
ax.plot(iter_lst, result_lst)
ax.set_xlabel('iter')
ax.set_ylabel('psnr')
ax.grid(axis='both')
plt.tight_layout()
fig.savefig('valid_curve.png')
plt.show()
