
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

