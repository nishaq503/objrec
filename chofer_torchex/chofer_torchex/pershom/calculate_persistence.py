# todo delete?

from torch import Tensor

from .pershom_backend import find_merge_pairings, merge_columns_, read_barcodes


def calculate_persistence(
        descending_sorted_boundary_array: Tensor,
        simplex_dimension: Tensor,
        max_dimension: int,
        max_pairs: int = -1
) -> [Tensor]:
    # This function is currently buggy. See pershom_dev/bug_1.py for reproduction. Cause could be the ATen backend

    iterations = 0
    while True:
        # TODO: FIX: synchronization issue appears here
        pivots = descending_sorted_boundary_array[:, 0].unsqueeze(1).contiguous()

        try:
            merge_pairings = find_merge_pairings(pivots, max_pairs)

        except Exception as ex:
            print(ex)

            print("Reached end of reduction after {} iterations".format(iterations))
            break

        merge_columns_(descending_sorted_boundary_array, merge_pairings)
        iterations += 1

    return read_barcodes(pivots, simplex_dimension, max_dimension)
