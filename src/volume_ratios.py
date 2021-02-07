import os
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyclam import Manifold, criterion, Cluster

from src.toy_shapes import SHAPES, plot_shape
from src.utils import SHAPES_DIR, PLOTS_DIR


def volume_ratios(data: np.ndarray, filename: str) -> pd.DataFrame:
    if os.path.exists(filename):
        volumes_df = pd.read_csv(filename)
        volumes_df.fillna('', inplace=True)
    else:
        # Create manifold from data
        manifold = Manifold(data, 'euclidean').build(
            criterion.MaxDepth(16),
        )

        # get volumes of all clusters in the manifold
        volumes: Dict[Cluster, float] = {
            cluster: cluster.radius ** 3
            for layer in manifold.layers
            for cluster in layer.clusters
        }
        clusters: List[Cluster] = list(sorted(list(volumes.keys())))
        clusters_enumerations: Dict[Cluster, int] = {c: i for i, c in enumerate(clusters)}

        # Initialize table for volume ratios
        ratios = np.zeros(shape=(len(volumes), manifold.depth + 1), dtype=np.float32)
        for c, i in clusters_enumerations.items():
            ratios[i][c.depth] = c.radius ** 3

        # populate table with correct ratios
        for graph in manifold.graphs:
            for cluster in graph.clusters:
                for g in manifold.graphs[cluster.depth + 1:]:
                    children = [c for c in g if cluster.name == c.name[:cluster.depth]]
                    total_volume = sum((volumes[c] for c in children)) + np.finfo(np.float32).eps
                    ratios[clusters_enumerations[cluster]][g.depth] = ratios[clusters_enumerations[cluster]][cluster.depth] / total_volume
                ratios[clusters_enumerations[cluster]][cluster.depth] = 0.

        # write a .csv of ratios
        volumes_df = pd.DataFrame(data=ratios)
        volumes_df['cluster_names'] = [cluster.name for cluster in clusters]
        volumes_df.to_csv(filename, index=False)

    return volumes_df


def _plot_ratios(volumes_df: pd.DataFrame, filename: str):
    names = [str(int(n)) if n != 'root' else '' for n in volumes_df['cluster_names']]
    del volumes_df['cluster_names']
    ratios = np.asarray(volumes_df.values, dtype=np.float32)

    plt.close('all')
    fig = plt.figure(figsize=(8, 8), dpi=300)
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


def plot_ratios():
    os.makedirs(SHAPES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for shape in SHAPES:
        np.random.seed(42)

        points = SHAPES[shape]()
        plot_shape(points)

        _plot_ratios(
            volume_ratios(points.T, os.path.join(SHAPES_DIR, f'{shape}.csv')),
            os.path.join(PLOTS_DIR, f'{shape}.png')
        )
