import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SHAPES_DIR = os.path.join(SRC_DIR, 'shapes')
DATA_DIR = os.path.join(SRC_DIR, 'data')
PLOTS_DIR = os.path.join(SRC_DIR, 'plots')
BARCODES_DIR = os.path.join(DATA_DIR, 'barcodes')


def increment_filename(filename: str) -> str:
    # expected filename format is {directory}/{name}__{number}.{extension}
    while os.path.exists(filename):
        split_name = filename.split('.')
        dir_name_num, extension = split_name[0], split_name[1]
        split_name = dir_name_num.split('__')
        dir_name, number = split_name[0], int(split_name[1])
        number += 1
        filename = f'{dir_name}__{number}.{extension}'

    return filename


def hist_plot(x: np.array, bins: Optional[int], log: bool, xlabel: str, title: str, filename: str):
    fig = plt.figure(figsize=(16, 10), dpi=200)
    fig.add_subplot(111)
    plt.hist(x, bins=bins, range=(0, max(x)), log=log)

    plt.xlabel(xlabel)
    plt.ylabel('counts')
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    return
