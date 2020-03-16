import os
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyclam import Manifold, criterion, Cluster

from src.toy_shapes import SHAPES, plot_shape

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BUILD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
PLOTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))


def volume_ratios(data: np.ndarray, filename: str) -> pd.DataFrame:
    if os.path.exists(filename):
        volumes_df = pd.read_csv(filename)
        volumes_df.fillna(0, inplace=True)
    else:
        # Create manifold from data
        manifold = Manifold(data, 'euclidean').build_tree(criterion.MaxDepth(16), criterion.MinPoints(8))
        # manifold = manifold.build_tree(criterion.MaxDepth(20), criterion.MinPoints(2))

        # get volumes of all clusters in the manifold
        volumes: Dict[Cluster, float] = {c: c.radius ** 3 for g in manifold.graphs for c in g.clusters}
        names = list(sorted([c.name for c in volumes], key=len))
        clusters = {manifold.select(n): i for i, n in enumerate(names)}

        # Initialize table for volume ratios
        ratios = np.zeros(shape=(len(volumes), manifold.depth + 1), dtype=np.float32)
        for c, i in clusters.items():
            ratios[i][c.depth] = c.radius ** 3

        # populate table with correct ratios
        for graph in manifold.graphs:
            for cluster in graph.clusters:
                for g in manifold.graphs[cluster.depth + 1:]:
                    children = [c for c in g if cluster.name == c.name[:cluster.depth]]
                    total_volume = sum((volumes[c] for c in children)) + np.finfo(np.float32).eps
                    ratios[clusters[cluster]][g.depth] = ratios[clusters[cluster]][cluster.depth] / total_volume
                ratios[clusters[cluster]][cluster.depth] = 0.

        # write a .csv of ratios
        volumes_df = pd.DataFrame(data=ratios)
        volumes_df['cluster_names'] = names
        volumes_df.to_csv(filename, index=False)

    return volumes_df


def plot_ratios(volumes_df: pd.DataFrame, filename: str):
    names = [str(int(n)) for n in volumes_df['cluster_names']]
    del volumes_df['cluster_names']
    ratios = np.asarray(volumes_df.values, dtype=np.float32)

    plt.close('all')
    fig = plt.figure(figsize=(6, 6), dpi=512)
    fig.add_subplot(111)
    x = list(range(ratios.shape[1]))

    n = 7
    [plt.plot(x, row) for row in ratios[:n]]
    plt.legend(names[:n], loc='lower right')

    title = filename.split('/')[-1].split('.')[0]
    plt.title(f'{title}: volume-ratio vs depth')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
    plt.show()
    return


if __name__ == '__main__':
    os.makedirs(BUILD_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    for shape in SHAPES:
        np.random.seed(42)

        points = SHAPES[shape]()
        plot_shape(points)

        plot_ratios(
            volume_ratios(points.T, os.path.join(BUILD_PATH, f'{shape}.csv')),
            os.path.join(PLOTS_PATH, f'{shape}.png')
        )
