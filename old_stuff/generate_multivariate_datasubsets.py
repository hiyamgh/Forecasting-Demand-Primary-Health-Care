import pandas as pd
import sys
sys.path.append("../../..")
import os
print(os.getcwd())
from geopy.geocoders import Nominatim
from geopy import distance
from Code.timeseries import TimeSeries, get_timeseries_components, add_lags, difference_series
from helper_codes.lags_dicts import weekly_dfs_to_difference, biweekly_dfs_to_difference, weekly_lags, biweekly_lags, weekly_stationary_dfs, \
    weekly_lags_without_diff, biweekly_lags_without_diff
import glob
import pickle


def generate_dictionary():
    """
    generates a dictionary mapping between mohafazas of interest (akkar, bekaa, tripoli)
    and the places of vdc events
    :return: dictionary distances_dict_updated.pickle saved hard-coded
    """
    # take other references for mohafaza names by mohafaza cities
    # so that we dont get conflicting names in google maps
    mohafaza_cities = {'north': "Zgharta-Ehden",
                       'bikaa': "Zahle",
                       'akkar': "Al-Qoubaiyat"}

    df = pd.read_csv('input/vdc/vdc_grouped.csv')
    vdc_places = list(df.place_of_death.unique())
    distances = {}

    vdc_places.remove('Other Nationalities')
    vdc_places.remove('Unknown')
    geolocator = Nominatim(user_agent="DSProject", timeout=40)
    for place in vdc_places:
        if place == 'Damascus Suburbs':
            place = 'Damascus'
        distances[place] = {}
        for mohafaza, city in mohafaza_cities.items():
            #print(place, mohafaza)
            location1 = geolocator.geocode(city)
            location2 = geolocator.geocode(place)
            latlng1 = (location1.latitude, location1.longitude)
            latlng2 = (location2.latitude, location2.longitude)
            dist_km = distance.distance(latlng1, latlng2).km
            print('mohafaza: %s, syrian_city: %s\nmohafaza-latlng: %s, syrian_city_latlng: %s' % (city, place, str(latlng1), str(latlng2)))
            distances[place][mohafaza] = dist_km
    print(distances)
    dist_df = pd.DataFrame.from_dict(distances)

    if not os.path.exists('input/distance_dictionary/'):
        os.makedirs('input/distance_dictionary/')

    dist_df.transpose().to_csv('input/distance_dictionary/distances.csv')
    pickle_out = open("input/distance_dictionary/distances_dict_updated.pickle", "wb")
    pickle.dump(distances, pickle_out)
    pickle_out.close()


class TimeSeriesGeneratorMultivariate:
    """
    This class is specific to the 12 data subsets we will be using in our project (MoPH).
    This class will add all the external data needed to generate 12 multivariate_datasubsets time
    series data. It will add:
    1. The time series components: trend, seasonality, and residual to each data subset
    2. The most appropriate number of lags for each data subset.
    3. The weather data for each data subset
    4. The VDC data for each data subset
    """

    def __init__(self, df=None, down_sample='W-TUE'):
        """
        :param df: the dataframe.
        :param down_sample: data aggregation. Example: if 'W-TUE' then its a weekly aggregation. If '2W-TUE',
        its a biweekly aggregation. If 'M', its a monthly aggregation. By deault, its weekly ('W-TUE')
        """
        
        self.very_original_df = df
        self.df_count = pd.DataFrame(TimeSeries(data_frame=df, downsample=down_sample).series)
        self.df_count = self.df_count.rename(index=str, columns={"count": "demand"})
        self.df_formatted = self.df_count
        self.downsample = down_sample

        ''' load the dictionary for distances between large scale event and mohafaza '''
        pickle_in = open("input/distance_dictionary/distances_dict_updated.pickle", "rb")
        self.distances = pickle.load(pickle_in)
        
    def add_VDC(self, moh):

        """
        adds VDC ranks to the data subset
        :param downsample: W-TUE or 2W-TUE
        :param moh: name of the governoarate (mohafaza)
        :return

            1. get df of old vdc (without ranking)
            2. get the ranked vdc (either weekly or bi-weekly)
            3. add distance column to this old df
            4. down-sample old vdc df weekly/biweekly, agg distance by average
            5. merge old df with ranked vdc df
        """

        old_vdc = pd.read_csv('input/vdc/vdc_grouped.csv')
        df = self.df_formatted
        vdc_ranked = pd.read_csv('output/vdc/civilians_rank.csv')
        vdc_ranked = vdc_ranked[['date', 'civ_quantile_label']]
        vdc_ranked = vdc_ranked.rename(columns={'civ_quantile_label': 'civilians_rank'})
        vdc_ranked = vdc_ranked.set_index(pd.to_datetime(vdc_ranked['date']))
        distances = []
        for i, row in old_vdc.iterrows():
            event = row['place_of_death']
            if moh == 'Tripoli':
                moh = 'north'
            if event == 'Other Nationalities' or event == 'Unknown' or event == 'Damascus Suburbs':
                dist = self.distances.get('Damascus').get(moh)
            else:
                dist = self.distances.get(event).get(moh)
            distances.append(dist)

        old_vdc['distance'] = distances
        old_vdc = old_vdc.set_index('date_of_death')
        old_vdc.index = pd.to_datetime(old_vdc.index)
        old_vdc = old_vdc[['distance']]
        vdc_ranked = vdc_ranked[['civilians_rank']]
        old_vdc = old_vdc.resample(self.downsample).mean()
        vdc_ranked = vdc_ranked.merge(old_vdc, how='inner', left_index=True, right_index=True)
        df = df.merge(vdc_ranked, how='inner', left_index=True, right_index=True)
        return df

    def add_weather_data(self, df,):

        """"
        adds weather data to the data subset
        :param df_orig: the data subset passed
        :param downsample: W-TUE or 2W-TUE
        :return

            1. set 'date' to be the index of weather
            2. down-sample weather
            3. merge df_orig with weather
        """

        # step 1
        weather = pd.read_csv('output/weather/Weather.csv')
        weather['Date'] = pd.to_datetime(weather['Date'])

        weather = weather.set_index('Date')

        weather['AverageWindSpeed'] = weather[['MinWindSpeed', 'MaxWindSpeed']].mean(axis=1)

        # step 2
        weather = weather.resample(self.downsample).agg({
            'AverageTemp': 'mean',
            'AverageWindSpeed': 'mean',
            'Precipitation': 'sum'
        })

        # step 3
        df = df.merge(weather, how='inner', left_index=True, right_index=True)
        return df


def generate_df_name(file_name):
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']
    for service in services:
        for mohafaza in mohafazas:
            if service in file_name and mohafaza in file_name:
                dir = service + '_' + mohafaza
                return dir, service, mohafaza


def generate_multivariate(down_sampling, output_folder, with_differencing=False):
    # generate_dictionary()
    fnames = glob.glob('output/Faour_datasubsets/*.csv')
    for f in fnames:
        df = pd.read_csv(f)
        dir, service_name, mohafaza = generate_df_name(f)
        tgm = TimeSeriesGeneratorMultivariate(df, down_sample=down_sampling)

        ''' adding VDC external data '''
        # if we are aggregating weekly

        multivariate_df = tgm.add_VDC(mohafaza)

        multivariate_df = tgm.add_weather_data(multivariate_df)

        if with_differencing:
            # difference the series
            ''' difference the data THEN add lags (so the lags are those of the differenced data) '''
            if dir in weekly_dfs_to_difference:
                print('...differencing weekly % s in %s' % (service_name, mohafaza))
                multivariate_df = difference_series(multivariate_df, 'demand', interval=1)

        ''' add lags '''
        if with_differencing:
            # add the lags that correspond to the 'differenced' according to their acf plot
            if down_sampling == 'W-TUE':
                lags = weekly_lags.get(dir)
            else:
                lags = biweekly_lags.get(dir)
        else:
            # add the lags that correspond to the 'UN-differenced' according to their acf plot
            if down_sampling == 'W-TUE':
                lags = weekly_lags_without_diff.get(dir)
            else:
                lags = biweekly_lags_without_diff.get(dir)
        multivariate_df = add_lags(df=multivariate_df, target_variable='demand', nb_lags=lags, col_prefix='w_{t-', col_suffix='}')

        # drop na values of lags
        multivariate_df = multivariate_df.dropna()

        # add trend and seasonality for the first lag
        multivariate_df = get_timeseries_components(multivariate_df, 'w_{t-1}', model='additive', freq=52)

        multivariate_df.index.name = 'date'
        df_name = dir + '.csv'

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        multivariate_df.to_csv(output_folder + df_name)

        if down_sampling == 'W-TUE':
            print(' --- Generated multivariate weekly df for %s in %s ---' % (service_name, mohafaza))
        else:
            print(' --- Generated multivariate bi-weekly df for %s in %s ---' % (service_name, mohafaza))


if __name__ == '__main__':
    # multivariate weekly, WITHOUT differencing
    generate_multivariate(down_sampling='W-TUE', output_folder='output/multivariate_datasubsets/', with_differencing=False)


