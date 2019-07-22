from collections import defaultdict

import numpy as np
import torch
from scipy.special import binom


def boundary_operator(simplex):
    s = tuple(simplex)

    if len(simplex) == 1:
        return ()

    else:
        return [s[:i] + s[(i + 1):] for i in range(len(s))]


def random_simplicial_complex(*args):
    simplexes_by_dim = defaultdict(set)

    vertex_ids = np.array((range(args[0])))

    vertices = [(i,) for i in vertex_ids]
    simplexes_by_dim[0] = set(vertices)

    for dim_i, n_simplexes in enumerate(args[1:], start=1):

        n_simplexes = min(n_simplexes, int(binom(len(vertices), dim_i + 1)))
        while len(simplexes_by_dim[dim_i]) != n_simplexes:
            chosen_vertices = np.random.choice(vertex_ids, replace=False, size=dim_i + 1)
            simplex = tuple(sorted(chosen_vertices))

            if simplex not in simplexes_by_dim[dim_i]:
                simplexes_by_dim[dim_i].add(simplex)

    for dim_i in sorted(simplexes_by_dim, reverse=True):
        if dim_i == 0:
            break

        for s in simplexes_by_dim[dim_i]:
            for boundary_s in boundary_operator(s):
                if boundary_s not in simplexes_by_dim[dim_i - 1]:
                    simplexes_by_dim[dim_i - 1].add(boundary_s)

    sp = []
    for dim_i in range(len(args)):
        sp += list(simplexes_by_dim[dim_i])

    return sp


def descending_sorted_boundary_array_from_filtrated_sp(filtrated_sp,
                                                       dtype=torch.int32,
                                                       resize_factor=2):
    simplex_to_ordering_position = {s: i for i, s in enumerate(filtrated_sp)}

    max_boundary_size = max(len(s) for s in filtrated_sp)
    n_cols = len(filtrated_sp)
    n_rows = resize_factor * max_boundary_size

    bm = torch.empty(size=(n_rows, n_cols),
                     dtype=dtype)
    bm.fill_(-1)

    col_to_dim = torch.empty(size=(n_cols,),
                             dtype=dtype)

    for col_i, s in enumerate(filtrated_sp):
        boundary = boundary_operator(s)
        orderings_of_boundaries = sorted((simplex_to_ordering_position[b] for b in boundary),
                                         reverse=True)

        col_to_dim[col_i] = len(s) - 1

        for row_i, entry in enumerate(orderings_of_boundaries):
            bm[row_i, col_i] = entry

    # boundary array is delivered in column first order for efficiency when merging
    bm = bm.transpose_(0, 1)

    return bm, col_to_dim
