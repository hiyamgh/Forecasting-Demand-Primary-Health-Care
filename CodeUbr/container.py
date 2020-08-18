import numpy as np
import itertools

'''
    This file will contain
    * the collection of different numpy array that will resemble the relevance matrices 
    we want to test in our data 
    * the different combinations of values for looping over them (thresholds, distances, methods, with/without
    relevance matrices)
'''

# the relevance matrices
# rel_mat1 = np.array([[100, 0, 0], [390, 1, 0], [400, 1, 0]])
# rel_mat2 = np.array([[100, 0, 0], [380, 1, 0], [400, 1, 0]])
# rel_mat3 = np.array([[100, 0, 0], [350, 1, 0], [400, 1, 0]])
# rel_mat4 = np.array([[100, 0, 0], [300, 1, 0], [400, 1, 0]])
# rel_mat5 = np.array([[100, 0, 0], [250, 1, 0], [400, 1, 0]])

rel_mat1 = np.array([[100, 0, 0], [200, 0, 0], [390, 1, 0]])
rel_mat2 = np.array([[100, 0, 0], [200, 0, 0], [380, 1, 0]])
rel_mat3 = np.array([[100, 0, 0], [200, 0, 0], [350, 1, 0]])
rel_mat4 = np.array([[100, 0, 0], [200, 0, 0], [300, 1, 0]])
rel_mat5 = np.array([[100, 0, 0], [200, 0, 0], [250, 1, 0]])


# dictionary for creating a unique name for each relevance matrix
relmat_dict = {
    'rel_mat1': rel_mat1,
    'rel_mat2': rel_mat2,
    'rel_mat3': rel_mat3,
    'rel_mat4': rel_mat4,
    'rel_mat5': rel_mat5,
}

# different values for: threshold, distance, and methods
relmats = list(relmat_dict.keys())

# for usage in combination1 - no relevance matrices
thresholds = [0.95, 0.9, 0.8]
distances = ["Manhattan", "Euclidean"]

# for usage in combination2 - with relevance marices
threshs = [[0.95], [0.9], [0.8]]
dists = [['Manhattan'], ['Euclidean']]

methods1 = ["smogn"]
methods2 = ["NoSmote"]


# define paths to the different input datasets
path_all_columns = '../input/all_without_date/collated/all_columns/'
path_all_columns_minus_weather = '../input/all_without_date/collated/all_columns_minus_weather/'
path_all_columns_minus_weather_minus_lags = '../input/all_without_date/collated/all_columns_minus_weather_minus_lags/'
path_all_columns_minus_weather_minus_vdc = '../input/all_without_date/collated/all_columns_minus_weather_minus_vdc/'
path_all_columns_minus_weather_minus_civilians = '../input/all_without_date/collated/all_columns_minus_weather_minus_civilians/'
path_all_columns_minus_weather_minus_distance = '../input/all_without_date/collated/all_columns_minus_weather_minus_distance/'
path_all_columns_minus_weather_minus_lags_minus_distance = '../input/all_without_date/collated/all_columns_minus_weather_minus_lags_minus_distance/'
path_all_columns_minus_weather_minus_lags_minus_civilians = '../input/all_without_date/collated/all_columns_minus_weather_minus_lags_minus_civilians/'
path_univariate = '../input/all_without_date/collated/univariate/'


datasets = {
    'all_columns': path_all_columns,
    'all_columns_minus_weather': path_all_columns_minus_weather,
    'all_columns_minus_weather_minus_lags': path_all_columns_minus_weather_minus_lags,
    'all_columns_minus_weather_minus_vdc': path_all_columns_minus_weather_minus_vdc,
    'all_columns_minus_weather_minus_civilians': path_all_columns_minus_weather_minus_civilians,
    'all_columns_minus_weather_minus_distance': path_all_columns_minus_weather_minus_distance,
    'all_columns_minus_weather_minus_lags_minus_distance': path_all_columns_minus_weather_minus_lags_minus_distance,
    'all_columns_minus_weather_minus_lags_minus_civilians': path_all_columns_minus_weather_minus_lags_minus_civilians,
    'univariate': path_univariate
}

# define scaling boundaries
scaling = {
    'all_columns': (0, 12),
    'all_columns_minus_weather': (0, 9),
    'all_columns_minus_weather_minus_lags': (0, 2),
    'all_columns_minus_weather_minus_vdc': (0, 7),
    'all_columns_minus_weather_minus_civilians': (0, 8),
    'all_columns_minus_weather_minus_distance': (0, 8),
    'all_columns_minus_weather_minus_lags_minus_civilians': (0, 1),
    'all_columns_minus_weather_minus_lags_minus_distance': (0, 1),
    'univariate': (0, 5)
}


def get_bool_method(method):
    ''' function that gets the name of the method and returns a vector of boolean values '''
    # smogn=True, rand_under=False, smoter=False, gn=False, nosmote=False
    if method == 'smogn':
        return [True, False, False, False, False]
    elif method == 'smoter':
        return [False, False, True, False, False]
    elif method == 'gn':
        return [False, False, False, True, False]
    elif method == 'rand_under':
        return [False, True, False, False, False]
    else:
        return [False, False, False, False, True]


def get_rel_mat(rel_mat_name):
    ''' function that gets the appropriate relevance metrix from the ones we defined above
    by taking its name '''
    return relmat_dict[rel_mat_name]

# create 2 combinations, the first with all possible values and the second
# combining only 'NoSmote' with different values of thresholds
# because threshold is the only relevant thing


# without relevance matrices
combination1 = list(itertools.product(thresholds, distances, methods1))

combination2 = list(itertools.product(thresholds, distances, methods1, relmats))

# No Oversampling at all
combination3 = list(itertools.product(thresholds, methods2))


if __name__ == '__main__':
    print()
