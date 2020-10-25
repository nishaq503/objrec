import numpy as np
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from pyclam import datasets as pyclam_datasets


def sphere(
        radius: float = 5.,
        noise: float = 1e-2,
        num_points: int = 10**3,
) -> np.ndarray:
    data: np.ndarray = np.random.randn(3, num_points)
    data /= np.linalg.norm(data, axis=0)
    data *= radius

    return data + (np.random.randn(3, num_points) * noise)


def torus(
        radius: float = 5.,
        noise: float = 1e-2,
        num_points: int = 10**3,
) -> np.ndarray:
    return np.stack(pyclam_datasets.generate_torus(n=num_points, r_torus=radius, noise=noise))


def klein_bottle(
        radius: float = 5.,
        noise: float = 1e-2,
        num_points: int = 10**3,
) -> np.ndarray:
    u, v = np.random.rand(num_points), np.random.rand(num_points)
    u, v = u * np.pi, v * 2 * np.pi
    cu, su, cv, sv = np.cos(u), np.sin(u), np.cos(v), np.sin(v)
    x = ((0 - 2 / 15) * cu * (3 * cv - 30 * su + 90 * (cu ** 4) * su
                              - 60 * (cu ** 6) * su + 5 * cu * cv * su)
         + (np.random.randn(num_points) * noise))
    y = ((0 - 1 / 15) * su * (3 * cu - 3 * (cu ** 2) * cv - 48 * (cu ** 4) * cv + 48 * (cu ** 6) * cv
                              - 60 * su + 5 * cu * cv * su - 5 * (cu ** 3) * cv * su
                              - 80 * (cu ** 5) * cv * su + 80 * (cu ** 7) * cv * su)
         + (np.random.randn(num_points) * noise))
    z = ((2 / 15) * (3 + 5 * cu * su) * sv + (np.random.randn(num_points) * noise))
    x, y, z = x - np.median(x), y - np.median(y), z - np.median(z)

    return np.stack((x, y, z)) * radius


def plot_shape(data: np.ndarray):
    x, y, z = data[0], data[1], data[2]

    plt.close('all')
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')

    limits = [np.min(data), np.max(data)]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)

    plt.grid(b=None)
    plt.axis('off')
    plt.show()
    return


SHAPES = {
    'sphere': sphere,
    'torus': torus,
    'klein_bottle': klein_bottle,
}
