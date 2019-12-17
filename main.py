import os
from typing import List

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
        variance: float = 10,
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
    noise = np.random.randn(3, 10 ** num_points) / variance
    data *= radius
    data += noise
    df = pd.DataFrame(data.T, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)
    return


def generate_torus(
        filename: str = None,
        num_points: int = 4,
        variance: float = 10,
        radius_c: float = 4,
        radius_a: float = 1,
):
    if not filename:
        suffix = '_'.join(map(str, [num_points, variance, radius_c, radius_a]))
        filename = f'shapes/torus_{suffix}.csv'
    u, v = np.random.rand(10 ** num_points), np.random.rand(10 ** num_points)
    u, v = u * 2 * np.pi, v * 2 * np.pi
    x = (radius_c + radius_a * np.cos(v)) * np.cos(u) + (np.random.randn(10 ** num_points) / variance)
    y = (radius_c + radius_a * np.cos(v)) * np.sin(u) + (np.random.randn(10 ** num_points) / variance)
    z = radius_a * np.sin(v) + (np.random.randn(10 ** num_points) / variance)

    df = pd.DataFrame(columns=['x', 'y', 'z'])
    df['x'], df['y'], df['z'] = x, y, z
    df.to_csv(filename, index=False)
    return


def generate_klein_bottle(
        filename: str = None,
        num_points: int = 4,
        variance: float = 10,
):
    if not filename:
        suffix = '_'.join(map(str, [num_points, variance]))
        filename = f'shapes/klein_bottle_{suffix}.csv'
    u, v = np.random.rand(10 ** num_points), np.random.rand(10 ** num_points)
    u, v = u * np.pi, v * 2 * np.pi
    cu, su, cv, sv = np.cos(u), np.sin(u), np.cos(v), np.sin(v)
    x = ((0 - 2 / 15) * cu * (3 * cv - 30 * su + 90 * (cu ** 4) * su
                              - 60 * (cu ** 6) * su + 5 * cu * cv * su)
         + (np.random.randn(10 ** num_points) / variance))
    y = ((0 - 1 / 15) * su * (3 * cu - 3 * (cu ** 2) * cv - 48 * (cu ** 4) * cv + 48 * (cu ** 6) * cv
                              - 60 * su + 5 * cu * cv * su - 5 * (cu ** 3) * cv * su
                              - 80 * (cu ** 5) * cv * su + 80 * (cu ** 7) * cv * su)
         + (np.random.randn(10 ** num_points) / variance))
    z = ((2 / 15) * (3 + 5 * cu * su) * sv
         + (np.random.randn(10 ** num_points) / variance))
    x, y, z = x - np.median(x), y - np.median(y), z - np.median(z)

    df = pd.DataFrame(columns=['x', 'y', 'z'])
    df['x'], df['y'], df['z'] = x, y, z
    df.to_csv(filename, index=False)
    return


def plot_points(filename: str, limits: List[int]):
    data = pd.read_csv(filename, dtype={'x': float,
                                        'y': float,
                                        'z': float, })
    x = data['x'].values
    y = data['y'].values
    z = data['z'].values

    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    # plt.axis('off')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)
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


def get_volume_ratios(filename: str, depth=15):
    np.random.seed(42)
    search_object = get_clusters(filename, depth)
    volumes_dict = search_object.get_info_dict(lambda node, *args: node.radius ** 3)
    # print(search_object.root.radius)
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

    volumes_df = pd.DataFrame(data=volume_ratios, dtype=float)
    volumes_df['cluster_name'] = cluster_names
    volumes_df.to_csv(f'volumes/{filename.split("/")[1]}', index=False)
    return volume_ratios, search_object


if __name__ == '__main__':
    np.random.seed(42)
    # n_, v_, r_, x_, y_, z_ = 4, 10**5, 1, 0, 0, 0
    # file_name = f'shapes/noisy_sphere_{n_}_{v_}_{r_}_{x_}_{y_}_{z_}.csv'
    # generate_sphere(
    #     filename=file_name,
    #     num_points=n_,
    #     variance=v_,
    #     radius=r_,
    #     x_center=x_,
    #     y_center=y_,
    #     z_center=z_,
    # )
    # n_, v_, rc_, ra_ = 4, 1000, 20, 10
    # file_name = f'shapes/noisy_torus_{n_}_{v_}_{rc_}_{ra_}.csv'
    # generate_torus(
    #     filename=file_name,
    #     num_points=n_,
    #     variance=v_,
    #     radius_c=rc_,
    #     radius_a=ra_,
    # )
    n_, v_ = 4, 1000
    file_name = f'shapes/noisy_klein_bottle_{n_}_{v_}.csv'
    generate_klein_bottle(
        filename=file_name,
        num_points=n_,
        variance=v_,
    )
    plot_points(filename=file_name, limits=[-2, 2])
    # search_object_ = get_clusters(file_name, depth=20)
    # volume_ratios_ = get_volume_ratios(file_name, depth=20)
    # [print(i_, ':', line_) for i_, line_ in enumerate(volume_ratios_[:len(volume_ratios_) // 2])]
