import os
from typing import List

import numpy as np
from pyclam import Manifold

from src.toy_shapes import SHAPES
from src.utils import DATA_PATH, SHAPES_PATH


def key(i: int, j: int) -> int:
    if i == j:
        raise ValueError(f'\'i\' must not equal \'j\' for key. Must look up for distinct points')
    elif i < j:
        i, j = j, i
    return (i * (i - 1)) // 2 + j


def get_persistent_components(data: np.array) -> np.array:
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


def create_training_data(n: int = 10**3, m: int = 10):
    # N = 3 * (3 + 3) * m * m = 18.m^2  instances
    # 3 shapes
    # m distortions each in x, y, z, tay, pitch, and roll
    # m shuffles of each instance of shape above

    for shape in SHAPES:
        count = 0
        for distortion in range(6):
            for _ in range(m):
                data = SHAPES[shape](num_points=n)
                if distortion < 3:  # x, y, or z
                    scale = np.random.sample() * 1.5 + .5
                    data[distortion] = data[distortion] * scale
                else:
                    angle = np.random.sample()
                    if distortion == 3:  # yaw, rotation about z-axis
                        matrix = np.asarray([
                            [np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1],
                        ])
                    elif distortion == 4:  # pitch, rotation about y-axis
                        matrix = np.asarray([
                            [np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)],
                        ])
                    else:  # roll, rotation about x-axis
                        matrix = np.asarray([
                            [1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)],
                        ])
                    data = np.matmul(matrix, data)
                data = data.T
                csv_name: str = f'{shape}-{count}.csv'
                count += 1
                np.savetxt(os.path.join(SHAPES_PATH, csv_name), data, fmt='%.12f', delimiter=',')

                persistence_vectors: np.array = get_persistent_components(data)
                # noinspection PyTypeChecker
                np.savetxt(os.path.join(DATA_PATH, csv_name), persistence_vectors, fmt='%d', delimiter=',')
    return


if __name__ == '__main__':
    np.random.seed(42)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(SHAPES_PATH, exist_ok=True)
    create_training_data(n=1000, m=10)
