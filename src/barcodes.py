import os
import time
from typing import Dict, Tuple, Set, List, Optional

import click
import numpy as np
from matplotlib import pyplot as plt
from pyclam import Cluster, Manifold

from toy_shapes import SHAPES
from utils import PLOTS_PATH

# Cluster -> (birth-radius, death-radius)
Barcodes = Dict[Cluster, Tuple[float, float]]


def create_barcodes(data: np.array, *, steps: Optional[int] = 10**3, normalize: bool = False) -> Barcodes:
    manifold: Manifold = Manifold(data, 'euclidean').build()
    thresholds: np.array = np.linspace(start=manifold.root.radius * (steps - 1) / steps, stop=0, num=steps)
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

    if normalize:  # normalize radii to [0, 1] range.
        factor = manifold.root.radius
        barcodes = {c: (b / factor, d / factor) for c, (b, d) in barcodes.items()}

    return barcodes


def plot_barcodes(full_barcodes: Barcodes, filename: str, *, merge: Optional[int] = None):
    cardinalities: List[int] = list(sorted({cluster.cardinality for cluster in full_barcodes}))
    barcodes_by_cardinality: Dict[int, Barcodes] = {cardinality: dict() for cardinality in cardinalities}
    [barcodes_by_cardinality[cluster.cardinality].update({cluster: lifetime})
     for cluster, lifetime in full_barcodes.items()]

    if merge is not None:
        # Merges all barcodes for clusters with cardinality greater than 'merge'
        high_cardinalities = [v for c, v in barcodes_by_cardinality.items() if c >= merge]
        if len(high_cardinalities) > 0:
            [high_cardinalities[0].update(h) for h in high_cardinalities[1:]]
            barcodes_by_cardinality = {c: v for c, v in barcodes_by_cardinality.items() if c < merge}
            barcodes_by_cardinality[merge] = high_cardinalities[0]

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


@click.group()
def cli():
    pass


@cli.command()
@click.argument('num_points', type=int, default=10**3)
@click.argument('steps', type=int, default=10**3)
def main(num_points: int, steps: int):
    for shape in SHAPES:
        start_time: float = time.time()
        barcodes = create_barcodes(SHAPES[shape](num_points=num_points).T, steps=steps, normalize=True)
        print(f'It took {time.time() - start_time:.2f} seconds for a {shape} with {num_points} points and {steps} resolution.')
        plot_barcodes(barcodes, os.path.join(PLOTS_PATH, f'barcodes-{shape}.png'), merge=4)
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_PATH, exist_ok=True)
    main()
