import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main(log_fp, target_criteria, save_dir, if_timestamp=True):
    save_dir = save_dir.resolve()

    log_fp = Path(log_fp).resolve()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_name = log_fp.parent.name
    if if_timestamp:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        save_fp = save_dir / f'valid_curve_{log_name}_{target_criteria}_{timestamp}.png'
    else:
        save_fp = save_dir / f'valid_curve_{log_name}_{target_criteria}.png'

    skip_lines = 0
    line_lst = log_fp.read_text().splitlines()[skip_lines:]

    iter_lst = []
    result_lst = []
    best_iter = -1
    best_val = -1
    for idx_line, line in enumerate(line_lst):
        if 'model is saved at' in line:
            model_path = Path(line[line.find('[') + 1: line.find(']')])
            pt_stem = model_path.stem
            iter_ = int(pt_stem.split('_')[-1])
            iter_lst.append(iter_)

            next_line = line_lst[idx_line + 1]
            pos_ = next_line.find(target_criteria)
            result = float(next_line[next_line.find('[', pos_) + 1: next_line.find(']', pos_)])
            result_lst.append(result)

            next_line = line_lst[idx_line + 2]
            pos_ = next_line.find(']]')
            best_iter = next_line[next_line.find('[[') + 2: pos_]
            if ',' not in best_iter:
                best_iter = int(best_iter)
            else:
                best_iter = int(best_iter[:best_iter.find(',')])
            best_val = float(next_line[next_line.find('[', pos_+2) + 1: next_line.find(']', pos_+2)])

    fig, ax = plt.subplots()
    ax.plot(iter_lst, result_lst)
    ax.plot([best_iter, best_iter], [min(result_lst), max(result_lst)], label=f'{best_iter}')
    ax.set_title(log_name)
    ax.set_xlabel('iter')
    ax.set_ylabel(target_criteria)
    ax.legend()
    ax.grid(axis='both')
    plt.tight_layout()
    fig.savefig(save_fp)
    # plt.show()


if __name__ == '__main__':
    import yaml

    with open('opt.yml', 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
        opts_dict = opts_dict['plot_curve_from_log']

    log_fp = opts_dict['log_fp']
    target_criteria = opts_dict['target_criteria']
    save_dir = Path(opts_dict['save_dir'] if opts_dict['save_dir'] is not None else './logs/')
    main(log_fp, target_criteria, save_dir)
