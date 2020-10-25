from typing import List

import numpy as np
from pyclam import Manifold

from src.toy_shapes import SHAPES


def key(i: int, j: int) -> int:
    if i == j:
        raise ValueError(f'i must not equal j for key. Must look up for distinct points')
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


def main():
    for shape in SHAPES:
        persistence_vectors = get_persistent_components(shape)
        print(f'\nshape: {shape}')
        for i, row in enumerate(persistence_vectors):
            components = np.unique(row)
            print(f'depth: {i + 1}, num_components: {len(components)}', components)
    return


if __name__ == '__main__':
    main()
