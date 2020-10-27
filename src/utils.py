import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

SHAPES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shapes'))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
PLOTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))


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
