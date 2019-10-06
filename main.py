import os

import numpy as np

from src.distance_functions import tf_calculate_pairwise_distances, tf_calculate_distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def check_df(df='sql2'):
    x = np.asfarray([[1, 2, 3], [4, 5, 6]])
    y = np.asfarray([3, 3, 3])
    
    print(tf_calculate_distance(x, y, df))
    print(tf_calculate_pairwise_distances(x, df))
    return


if __name__ == '__main__':
    check_df()
