import os
from typing import List

import numpy as np
from pyclam import Manifold

from src.toy_shapes import SHAPES
from src.utils import hist_plot, PLOTS_PATH


def key(i: int, j: int) -> int:
    if i == j:
        raise ValueError(f'\'i\' must not equal \'j\' for key. Must look up for distinct points')
    elif i < j:
        i, j = j, i
    return (i * (i - 1)) // 2 + j


def get_persistent_components(shape: str) -> np.ndarray:
    data = SHAPES[shape](num_points=10**3).T
    manifold: Manifold = Manifold(data, 'euclidean').build()
    num_cells = (len(data) * (len(data) - 1)) // 2
    persistence_vectors: np.array = np.zeros(shape=(manifold.depth + 1, num_cells), dtype=int)
    for depth, layer in enumerate(manifold.layers):
        for i, subgraph in enumerate(layer.subgraphs):
            points: List[int] = list()
            [points.extend(cluster.argpoints) for cluster in subgraph]
            for j, left in enumerate(points):
                for right in points[j + 1:]:
                    persistence_vectors[depth, key(left, right)] = i + 1
    return persistence_vectors


def depth_distribution(shape: str, filename: str):
    data = SHAPES[shape](num_points=10**3).T
    manifold: Manifold = Manifold(data, 'euclidean').build()
    depths: List[int] = [leaf.depth for leaf in manifold.layers[-1].clusters]
    hist_plot(depths, manifold.depth, False, 'depths', f'depths of leaves for {shape}', filename)
    return


def radii_distribution(shape: str, filename: str):
    data = SHAPES[shape](num_points=10**3).T
    manifold: Manifold = Manifold(data, 'euclidean').build()
    radii: List[float] = [
        cluster.radius for layer in manifold.layers
        for cluster in layer.clusters
        if cluster.radius > 0
    ]
    hist_plot(radii, 32, True, 'radius', f'radii of clusters in tree for {shape}', filename)
    return


def make_plots(option: str):
    if option == 'persistent-components':  # View Bit-Vectors for Shapes
        for shape in SHAPES:
            persistence_vectors = get_persistent_components(shape)
            print(f'\nshape: {shape}')
            for i, row in enumerate(persistence_vectors):
                components = np.unique(row)
                print(f'depth: {i + 1}, num_components: {len(components)}', components)
    elif option == 'depth-distributions':  # view depth distributions for leaves of Manifolds over shapes
        for shape in SHAPES:
            filename = os.path.join(PLOTS_PATH, f'{shape}-leaf-depths.png')
            depth_distribution(shape, filename)
    elif option == 'radii-distributions':  # view radii distributions for clusters in Manifolds over shapes
        for shape in SHAPES:
            filename = os.path.join(PLOTS_PATH, f'{shape}-radii.png')
            radii_distribution(shape, filename)
    else:
        raise ValueError(f'option {option} not implemented. Try:'
                         f'\'persistent-components\', '
                         f'\'depth-distributions\', or'
                         f'\'radii-distributions\'.')
    return
