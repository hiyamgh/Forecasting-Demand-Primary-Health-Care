import numpy as np

weather_dict = {
            'MinTemp': np.mean,
            'MaxTemp': np.mean,
            'MaxHumidity': np.mean,
            'MinHumidity': np.mean,
            'MaxWindSpeed': np.mean,
            'MinWindSpeed': np.mean,
            'Precipitation': np.mean,
}

socioeconomic_dict = {
    'Private Hospitals': np.mean,
    'Public Hospitals': np.mean,
    'public schools': np.mean,
    'free private schools': np.mean,
    'paid private schools': np.mean,
    'total including UNRWA': np.mean,
    'energy total': np.mean,
    'Foodstuff production': np.mean,
    'Building Materials': np.mean,
    'Metal and electrical products': np.mean,
    'Furniture and wood industry': np.mean,
    'Chemical industries': np.mean,
    'Rubber and plastic': np.mean,
    'Base metals industry': np.mean,
    'Publishing, printing and advertising': np.mean,
    'Miscellaneous tools and equipments production': np.mean,
    'Clothing production and fur tanning':np.mean,
    'Machinery production': np.mean,
    'Miscellaneous electrical products and tools': np.mean,
    'Mining and quarrying products': np.mean,
    'Textile products': np.mean,
    'Paper production': np.mean,
    'Total': np.mean,
    'unemployment rate': np.mean,
    'Tonnes/day': np.mean,
    'MTonnes/year': np.mean
}

sectarian_dict = {
    'Sunni': np.mean,
    'Shia': np.mean,
    'Druze': np.mean,
    'Aalawi': np.mean,
    'Maron': np.mean,
    'Cath': np.mean,
    'Orth': np.mean,
    'Prot': np.mean,
    'arminc': np.mean,
    'armino': np.mean
}

syrian_dict_sum = {
    'rank_casualties': np.sum,
    'rank/10': np.sum,
    'total_casualties': np.sum,
    'distance': np.mean
}

syrian_dict_avg = {
   'rank_casualties': np.mean,
    'rank/10': np.mean,
   'total_casualties': np.sum,
    'distance': np.mean
}
