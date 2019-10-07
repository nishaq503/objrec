import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from src import globals
from src.distance_functions import tf_calculate_distance, tf_calculate_pairwise_distances, numpy_calculate_distance
from src.search import Search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_sphere(
        filename: str = None,
        num_points: int = 4,
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
    data *= radius
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
    # noinspection PyTypeChecker
    search_object = Search(
        data=data,
        distance_function='l2',
        names_file='clusterings/sphere_names.csv',
        info_file='clusterings/sphere_info.csv',
    )
    return search_object


def get_all_children(parent_name, child_depth):
    old_names = [parent_name]
    new_names = []
    for _ in range(child_depth):
        [new_names.extend([f'{name}1', f'{name}2'])
         for name in old_names]
        old_names = new_names.copy()
        new_names = []
    return old_names


def get_volume_ratios(depth=15):
    np.random.seed(42)
    n, r, x, y, z = 4, 1, 0, 0, 0
    filename = f'shapes/sphere_{n}_{r}_{x}_{y}_{z}.csv'

    search_object = get_clusters(filename, depth)
    volumes_dict = search_object.get_info_dict(lambda node, *args: node.radius ** 3)
    cluster_names = list(sorted(list(volumes_dict.keys()), key=len))
    names_index = {name: i for i, name in enumerate(cluster_names)}

    volume_ratios = np.zeros(shape=(len(volumes_dict), depth + 1))
    for i, name in enumerate(cluster_names):
        volume_ratios[i][len(name)] = volumes_dict[name]
    for d in range(depth + 1):
        targets = [name for name in cluster_names if len(name) == d]
        for target in targets:
            for cd in range(d + 1, depth + 1):
                children = get_all_children(target, cd - d)
                children = [child for child in children if child in volumes_dict.keys()]
                if len(children) == 0:
                    continue
                volumes = [volumes_dict[child] for child in children]
                target_volume = sum(volumes)
                target_ratio = volume_ratios[names_index[target]][len(target)] / target_volume
                volume_ratios[names_index[target]][len(children[0])] = target_ratio
            volume_ratios[names_index[target]][len(target)] = 0
    return volume_ratios


if __name__ == '__main__':
    np.random.seed(42)
    n_, r_, x_, y_, z_ = 4, 10, 0, 0, 0
    file_name = f'shapes/sphere_{n_}_{r_}_{x_}_{y_}_{z_}.csv'
    # generate_sphere(filename=file_name, num_points=n_, radius=r_, x_center=x_, y_center=y_, z_center=z_)
    # plot_points(filename=file_name)
    # check_df('l2')
    # get_clusters(file_name, 15)
    volume_ratios_ = get_volume_ratios()
    # [print(i_, ':', line_) for i_, line_ in enumerate(volume_ratios_[:len(volume_ratios_) // 2])]
