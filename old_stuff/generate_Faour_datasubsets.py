import numpy as np
import os.path
from geopy.geocoders import Nominatim
from geopy import distance
from helper_codes.data_set import DataSet
from helper_codes.data_subset import DataSubset


class DataSubsetsGeneratorUnivariate:
    def __init__(self):
        # data_set_holder = data_set.DataSet()
        data_set_holder = DataSet()
        self.df = data_set_holder.load_data_frame()

    def generate_subsets_by_qada(self, service_array, nationality_array, qada_array):
        """
        a function that generates subsets of data, aggregated by the service types, nationalities, and qada of interest
        :param service_array: array of services we wish to include
        :param nationality_array: array of nationalities we wish to include
        :param qada_array: array of qada we wish to include
        :return: subsets generated (service x nationality x qada)
        """
        datasubsetter = DataSubset(self.df)
        bigdf = self.df[self.df['service_name'].isin(service_array) & self.df['nationality'].isin(nationality_array) & self.df['qada'].isin(qada_array)]
        dfs = datasubsetter.split_groups(bigdf, ['service_name', 'nationality', 'qada'])
        for d in dfs:
            i1 = d.columns.get_loc("service_name")
            i2 = d.columns.get_loc("nationality")
            i3 = d.columns.get_loc("qada")
            s = str(d.iloc[0, i1]) + "_" + str(d.iloc[0, i2]) + "_" + str(d.iloc[0, i3]) + ".csv"
            print("Generated file ", s)
            path = 'output/Faour_datasubsets/'
            if not os.path.exists(path):
                os.makedirs(path)
            d.to_csv(os.path.join(path, s))
        return dfs

    def generate_subsets_by_mohafaza(self, service_array, nationality_array, mohafaza_array):
        """
        a function that generates subsets of data, aggregated by the service types, nationalities, and mohafazas of interest
        :param service_array: array of services we wish to include
        :param nationality_array: array of nationalities we wish to include
        :param mohafaza_array: array of mohafazas we wish to include
        :return: subsets generated (service x nationality x mohafaza)
        """
        #Tripoli, ['akkar', 'bikaa'], ['Syria'], ['Pharmacy', 'General Medicine', 'Pediatrics', 'Gynaecology']
        #['General Medicine', 'Pharmacy', 'Pediatrics']
        #['Lebanon', 'Syria']
        #['beirut', 'north', 'south']
        datasubsetter = DataSubset(self.df)
        bigdf = self.df[self.df['service_name'].isin(service_array) & self.df['nationality'].isin(nationality_array) & self.df['mohafaza'].isin(mohafaza_array)]
        dfs = datasubsetter.split_groups(bigdf, ['service_name', 'nationality', 'mohafaza'])
        for d in dfs:
            i1 = d.columns.get_loc("service_name")
            i2 = d.columns.get_loc("nationality")
            i3 = d.columns.get_loc("mohafaza")
            s = str(d.iloc[0, i1]) + "_" + str(d.iloc[0, i2]) + "_" + str(d.iloc[0, i3]) + ".csv"
            print("Generated file ", s)
            path = 'output/Faour_datasubsets/'
            if not os.path.exists(path):
                os.makedirs(path)
            d.to_csv(os.path.join(path, s))
        return dfs


class DataSubsetsGenerator:
    """
    ['Aleppo' 'Deir Ezzor' 'Homs' 'Unknown' 'Daraa' 'Damascus Suburbs'
     'Quneitra' 'Tartous' 'Damascus' 'Idlib' 'Sweida' 'Other Nationalities'
     'Hama' 'Hasakeh' 'Lattakia' 'Raqqa']
    """
    def __init__(self, govs=None):
        """
        1. Gets the original MoPH data frame
        2. adds the additional datasets we used (socio economic, sectarian, public holidays, weather, syrian events)
        3. generate a dictionary of values, which maps each city (from the list of cities where the large scale events happened)
        to the governorates of interest, and each governorate f interest to two distance values (in km and in miles)
        4. use this dictionary in the add_distance(self, df) rather that querying the web over 900,000 records, which is more
        efficient
        """
        data_set_holder = DataSet()
        self.df = data_set_holder.load_data_frame()
        self.df = data_set_holder.add_external_data_old(self.df)
        self.mydict = {}
        geolocator = Nominatim(user_agent="DSProject", timeout=20)
        mylist = ['Aleppo', 'Deir Ezzor', 'Homs','Unknown', 'Daraa', 'Damascus Suburbs',
                  'Quneitra', 'Tartous', 'Damascus', 'Idlib', 'Sweida', 'Other Nationalities',
                   'Hama', 'Hasakeh', 'Lattakia', 'Raqqa']
        if govs is None:
            govs = ["Zgharta-Ehden", "Sidon","Beirut"]
        for place in mylist:
            if place in ['Other Nationalities', 'Unknown']:
                continue
            self.mydict[place] = {}
            for mz in govs:
                location1 = geolocator.geocode(mz)
                location2 = geolocator.geocode(place)
                latlng1 = (location1.latitude, location1.longitude)
                latlng2 = (location2.latitude, location2.longitude)
                dist_km = distance.distance(latlng1, latlng2).km
                dist_miles = distance.distance(latlng1, latlng2).miles
                if mz is 'Zgharta-Ehden':
                    m = 'north'
                if mz is 'Sidon':
                    m = 'south'
                if mz is 'Beirut':
                    m = 'beirut'
                self.mydict[place][m] = {'km': dist_km, 'miles': dist_miles}
        print(self.mydict)
        print('list of unique events', list(self.df.place_of_death.unique()))

    def generate_subsets_by_mohafaza(self, service_array, nationality_array, mohafaza_array, qada_array=None):
        """
        a function that enerates subsets of data, aggregated by the service types, nationalities, and mohafazas of interest
        :param service_array: array of services we wish to include
        :param nationality_array: array of nationalities we wish to include
        :return: subsets generated
        """
        if qada_array is None:
            datasubsetter = DataSubset(self.df)
            self.df = self.df[self.df['service_name'].isin(service_array) & self.df['nationality'].isin(nationality_array) & self.df['mohafaza'].isin(mohafaza_array)]
            dfs = datasubsetter.split_groups(self.df, ['service_name', 'nationality', 'mohafaza'])
            for d in dfs:
                d = self.add_distances(d)
                i1 = d.columns.get_loc("service_name")
                i2 = d.columns.get_loc("nationality")
                i3 = d.columns.get_loc("mohafaza")
                s = "UPDATED_" + str(d.iloc[0, i1]) + "_" + str(d.iloc[0, i2]) + "_" + str(d.iloc[0, i3]) + ".csv"
                print("Generated file ", s)
                d.to_csv(s)
            return dfs

    def add_distances(self, df):
        """
        a function that adds distances between the passed data frame and the place of the large
        scale event that happened in a certain day (expects that the syrian data be part of the passed
        data frame)
        :param df: data frame
        :return: data frame with two distance columns added (1. km, 2. miles)
        """
        distance_km = []
        distance_miles = []
        print('before', df.shape)
        df.dropna(subset=['place_of_death'], inplace=True)
        print('after', df.shape)
        for (i, row) in df.iterrows():
            mohafaza = str(row['mohafaza'])
            event = str(row['place_of_death'])
            if event in ("Other Nationalities", "Unknown"):
                distance_km.append(np.NaN)
                distance_miles.append(np.NaN)
            else:
                print("dictionary type", type(self.mydict))
                print('event is', event)
                print('mohafaza is', mohafaza)
                print('----------------------------------------')
                dist_km = self.mydict.get(event).get(mohafaza).get('km')
                dist_miles = self.mydict.get(event).get(mohafaza).get('miles')
                distance_km.append(dist_km)
                distance_miles.append(dist_miles)
        df['distance_km'] = distance_km
        df['distance_miles'] = distance_miles
        return df


if __name__ == '__main__':
    generator_univariate = DataSubsetsGeneratorUnivariate()

    ''' generates subsets for Akkar & Bekaa '''
    ll = generator_univariate.generate_subsets_by_mohafaza(['Pharmacy', 'General Medicine', 'Pediatrics', 'Gynaecology'], ['Syria'], ['akkar', 'bikaa'])

    ''' generate subsets for Tripoli '''
    ll2 = generator_univariate.generate_subsets_by_qada(service_array=['Pharmacy', 'General Medicine', 'Pediatrics', 'Gynaecology'], nationality_array=['Syria'], qada_array=['Tripoli'])
