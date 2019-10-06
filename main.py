import os
from pprint import pprint
from time import time
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from src import globals
from src.cluster import Cluster
from src.distance_functions import tf_calculate_distance, tf_calculate_pairwise_distances, numpy_calculate_distance
from src.search import Search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_sphere(
        filename: str = None,
        num_points: int = 3,
        radius: float = 1,
        x_center: float = 0,
        y_center: float = 0,
        z_center: float = 0,
):
    if not filename:
        suffix = '_'.join(map(str, [num_points, radius, x_center, y_center, z_center]))
        filename = f'shapes/sphere_{suffix}.csv'

    data = np.random.randn(3, 10 ** num_points)
    data /= np.linalg.norm(data, axis=0)
    df = pd.DataFrame(data.T, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)
    return


def plot_points(filename: str):
    data = pd.read_csv(filename, dtype={'x': float,
                                        'y': float,
                                        'z': float, })
    x = data['x'].values
    y = data['y'].values
    z = data['z'].values

    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    plt.axis('off')
    plt.grid(b=None)
    plt.show()
    return


def check_df(df='sql2'):
    data = pd.read_csv(
        'shapes/sphere_4_10_0_0_0.csv',
        dtype={
            'x': float,
            'y': float,
            'z': float,
        }
    )
    x = np.asfarray(data.values)
    print('numpy:\n', numpy_calculate_distance(x, x, df), '\n')
    print('tf:\n', tf_calculate_distance(x, x, df), '\n')
    print('tf:\n', tf_calculate_pairwise_distances(x, df), '\n')
    return


def get_clusters(filename: str, depth: int):
    data = pd.read_csv(
        filename,
        dtype={
            'x': float,
            'y': float,
            'z': float,
        }
    )
    data = np.asfarray(data.values)

    np.random.seed(42)
    globals.MAX_DEPTH = depth
    start = time()
    # noinspection PyTypeChecker
    search_object = Search(
        data=data,
        distance_function='l2',
        names_file='clusterings/sphere_names.csv',
        info_file='clusterings/sphere_info.csv',
    )
    end = time()

    cluster_dict: Dict[str: Cluster] = search_object.cluster_dict_
    num_leaves = sum([1 for _, c in cluster_dict.items() if c.left or c.right])
    print(f'time taken for {len(data)} points to depth {globals.MAX_DEPTH} was {end - start:.4f} seconds. '
          f'We got {num_leaves} leaf clusters.')

    volumes_dict: Dict[str: float] = search_object.get_info_dict(lambda node, *args: (4 / 3) * np.pi * (node.radius**3))
    pprint(volumes_dict)

    return search_object


if __name__ == '__main__':
    np.random.seed(42)
    n, r, x_, y_, z_ = 4, 10, 0, 0, 0
    file_name = f'shapes/sphere_{n}_{r}_{x_}_{y_}_{z_}.csv'
    # generate_sphere(filename=file_name, num_points=n, radius=r, x_center=x_, y_center=y_, z_center=z_)
    # plot_points(filename=file_name)
    # check_df('l2')
    get_clusters(file_name, 3)
