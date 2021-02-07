import os
from typing import List

from pyclam import Manifold

from src.toy_shapes import SHAPES
from src.utils import hist_plot, PLOTS_DIR


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
    if option == 'depth-distributions':  # view depth distributions for leaves of Manifolds over shapes
        for shape in SHAPES:
            filename = os.path.join(PLOTS_DIR, f'{shape}-leaf-depths.png')
            depth_distribution(shape, filename)
    elif option == 'radii-distributions':  # view radii distributions for clusters in Manifolds over shapes
        for shape in SHAPES:
            filename = os.path.join(PLOTS_DIR, f'{shape}-radii.png')
            radii_distribution(shape, filename)
    else:
        raise ValueError(f'option {option} not implemented. Try:'
                         f'\'depth-distributions\', or'
                         f'\'radii-distributions\'.')
    return
