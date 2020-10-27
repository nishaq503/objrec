import os
from typing import Dict, Tuple, Set, List

import numpy as np
from matplotlib import pyplot as plt
from pyclam import Cluster, Manifold

from src.toy_shapes import SHAPES
from src.utils import PLOTS_PATH

# Cluster, birth-radius, death-radius
Barcodes = Dict[Cluster, Tuple[float, float]]


def create_barcodes(data: np.array, steps: int) -> Barcodes:
    manifold: Manifold = Manifold(data, 'euclidean').build()
    step_size: float = -manifold.root.radius / steps
    thresholds: np.array = np.arange(start=manifold.root.radius - step_size, stop=0, step=step_size)
    barcodes: Barcodes = dict()
    living_clusters: Barcodes = {manifold.root: (-1, manifold.root.radius)}
    for threshold in thresholds:
        new_births: Set[Cluster] = set()
        dead_clusters: Set[Cluster] = {cluster for cluster in living_clusters if cluster.radius > threshold}
        while dead_clusters:
            cluster = dead_clusters.pop()
            death = living_clusters.pop(cluster)[1] if cluster in living_clusters else threshold
            barcodes[cluster] = threshold, death
            for child in cluster.children:
                if child.cardinality > 1:
                    (dead_clusters if child.radius > threshold else new_births).add(child)
                else:
                    barcodes[child] = (0, threshold)
        living_clusters.update({cluster: (-1, threshold) for cluster in new_births})
    return barcodes


def plot_barcodes(full_barcodes: Barcodes, filename: str):
    cardinalities: List[int] = list(sorted({cluster.cardinality for cluster in full_barcodes}))
    barcodes_by_cardinality: Dict[int, Barcodes] = {cardinality: dict() for cardinality in cardinalities}
    [barcodes_by_cardinality[cluster.cardinality].update({cluster: lifetime})
     for cluster, lifetime in full_barcodes.items()]
    max_radius: float = max((lifetime[1] for lifetime in full_barcodes.values()))

    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplot(111)

    height = 1
    y_ticks: Dict[int, str] = dict()
    for cardinality, barcodes in barcodes_by_cardinality.items():
        lifetimes = list(sorted(barcodes.values()))
        for birth, death in lifetimes:
            plt.plot([birth, death], [height, height], lw=0.8)
            height += 1
        y_ticks[height] = f'{cardinality}'

    ax.set_yticks(list(y_ticks.keys()))
    ax.set_yticklabels(list(y_ticks.values()))

    plt.xlabel('radius')
    plt.ylabel('cardinality')
    plt.title('barcodes')
    plt.savefig(filename, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    return


def main():
    for shape in SHAPES:
        barcodes = create_barcodes(SHAPES[shape](num_points=10**4).T, 10**4)
        plot_barcodes(barcodes, os.path.join(PLOTS_PATH, f'barcodes-{shape}.png'))
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_PATH, exist_ok=True)
    main()
