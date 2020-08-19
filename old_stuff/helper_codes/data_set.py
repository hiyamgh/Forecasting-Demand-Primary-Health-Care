import pandas as pd
import os.path
import numpy as np
import os
from old_stuff.helper_codes.translations import countries_dict, sex_dict, services_dict, mohafaza_dict, qada_dict
from geopy.geocoders import Nominatim
from geopy import distance

class DataSet:
    """
    A structure for loading, cleaning and subsetting data
    A data frame is loaded with only essential changes
    Additional cleaning and subsetting is done to copies of this, maintaining the original
    """

    def __init__(self):
        """
        self.abs_path: absolute path to the 'data_loading_aggregating'
        had to put this because when DataSet() is being called
        from another file, and it tries to access /output/FaourProcessed.csv
        it cannot reach it because its in another file and all links are static
        """
        # self.original_df = self.load_data_frame()
        self.abs_path = 'C:/Users/96171/Desktop/ministry_of_public_health/old_stuff/input/Faour/'
        self.output_path = 'C:/Users/96171/Desktop/ministry_of_public_health/old_stuff/output/Faour_Processed/'
        self.original_df = self.load_data_frame()

    def copy_df(self):
        """
        :return: a deep copy of the original dataframe, identical values but no references to the original
        """
        return self.original_df.copy(deep=True)

    def load_data_frame(self):
        """
        Load the data frame from .xlsx
        process dataframe, removing unecessary columns
        clean dataframe, get it in the correct format
        write to csv to skip this process next time
        :return: a processed dataframe
        """
        # print(os.getcwd())
        if os.path.exists(self.output_path+'FaourProcessed.csv'):
            df = pd.read_csv(self.output_path+'FaourProcessed.csv')
            print("loaded FaourProcessed.csv")
            self.dataframe_details(df)

        else:
            print("Preprocessed file 'FaourProcessed.csv' does not exist")
            print("Reading from xlsx and writing to csv to speed up repeated loading")

            df_sheet1 = pd.read_excel(self.abs_path + 'Faour.xlsx', sheet_name="Sheet1", encoding='utf-8')
            print("loaded Faour.xlsx sheet 1")

            df_sheet2 = pd.read_excel(self.abs_path + 'Faour.xlsx', sheet_name="Sheet2", encoding='utf-8', header=None)
            print("loaded Faour.xlsx sheet 2")

            df_sheet2.columns = df_sheet1.columns.tolist()
            df = pd.concat([df_sheet1, df_sheet2], sort=False)
            self.dataframe_details(df)

            df = self.data_preprocessing(df)
            self.dataframe_details(df)
            if not os.path.exists('output'):
                os.makedirs('output')
            df.to_csv(self.output_path + 'FaourProcessed.csv', encoding='utf-8', index=False)
            print("wrote data to FaourProcessed.csv")

        return df

    @staticmethod
    def add_public_holidays(df):
        try:
            holidays_dates = pd.read_csv("holidays.csv", parse_dates=["Date"])
        except FileNotFoundError:
            print("Public holidays csv not found")
        else:
            # convert holidays df to a dict
            date_to_encoding = pd.Series(holidays_dates.Encoding.values, index=holidays_dates.Date).to_dict()

            df.loc[~ df['date'].isin(holidays_dates['Date']), 'holiday'] = 0
            df.loc[df['date'].isin(holidays_dates['Date']), 'holiday'] = df[df['date'].isin(holidays_dates['Date'])]['date'].replace(date_to_encoding)
            pd.to_numeric(df.holiday)

    @staticmethod
    def add_time_columns(df):
        df['year'] = pd.DatetimeIndex(df['date']).year
        df['month'] = pd.DatetimeIndex(df['date']).month
        df['day'] = pd.DatetimeIndex(df['date']).day
        df['dayofweek'] = pd.DatetimeIndex(df['date']).dayofweek

    @staticmethod
    def add_sectarian_data(df):
        sectarian_df = pd.read_csv('sectarian_mean.csv')
        return pd.merge(df, sectarian_df,  how='left', left_on=['mohafaza'], right_on=['mohafaza'])


    @staticmethod
    def add_socioeconomic_data(df):
        socioeconomic_df = pd.read_csv('SOCIOECONOMICEDITED.csv')
        return pd.merge(df, socioeconomic_df,  how='left', left_on=['mohafaza', 'year'], right_on=['mohafaza', 'year'])

    @staticmethod
    def add_weather_data(df):
        weather_df = pd.read_csv("Weather.csv", parse_dates=["Date"])
        weather_df = weather_df.rename(index=str, columns={"Date": "date"})

        weather_series = weather_df.set_index('date')['MinTemp']
        min_weeks = weather_series.resample("W-MON").min()
        min_weeks = min_weeks[min_weeks <= 5.0]
        weather_series = weather_df.set_index('date')['MaxTemp']
        max_weeks = weather_series.resample("W-MON").max()
        max_weeks = max_weeks[max_weeks >= 35.0]

        # add temperature data to dataframe
        weather_df = weather_df.drop(columns=["Mohafaza","MaxHumidity", "MinHumidity", "MaxWindSpeed", "MinWindSpeed", "Precipitation"])
        df = pd.merge(df, weather_df, how='left', left_on=['date'], right_on=['date'])

        # mark extreme weeks
        min_days = [d - np.timedelta64(i, 'D') for d in min_weeks.index.values for i in range(7)]
        max_days = [d - np.timedelta64(i, 'D') for d in max_weeks.index.values for i in range(7)]
        df['extreme_cold'] = df.date.isin(min_days)
        df['extreme_heat'] = df.date.isin(max_days)

        return df

    @staticmethod
    def add_syrian_data(df):
        syrianevents = SyrianEvents()
        syrian = syrianevents.get_syrian_transformed()
        syrian['date'] = pd.to_datetime(syrian['date'])
        df = pd.merge(df, syrian, how='left', left_on=['date'], right_on=['date'])
        return df

        # syrian_df = pd.read_csv("SYRIANRANKED.csv")
        # syrian_df['date_of_death'] = pd.to_datetime(syrian_df['date_of_death'])
        # syrian_df = syrian_df[['date_of_death', 'nb_civilians', 'cause_of_death']]
        # syrian_df = syrian_df.groupby('date_of_death').agg({'nb_civilians': 'sum', 'cause_of_death': 'count'})
        ####TO BE CONTINUED TOMMOROW ######
        # df = pd.merge(df, syrian_df, how='left', left_on=['date'], right_on=['date'])
        # return df
        # syrian_df = syrian_df.rename(index=str, columns={"date_of_death": "date"})
        # df = pd.merge(df, syrian_df, how='left', left_on=['date'], right_on=['date'])
        # return df

    @staticmethod
    def add_syrian_data_old(df):
        syrian_df = pd.read_csv("SYRIANRANKED.csv")
        syrian_df['date_of_death'] = pd.to_datetime(syrian_df['date_of_death'])
        syrian_df = syrian_df.rename(index=str, columns={"date_of_death": "date"})
        df = pd.merge(df, syrian_df, how='left', left_on=['date'], right_on=['date'])
        return df


    @staticmethod
    def add_distances(df, mohafaza_name=None):
        # print("loading distances")
        # try:
        #     distances = pickle.load(open("leb_syria_distances.p", "rb"))
        # except Exception as e:
        #     print("cound not find leb_syria_distances.p, creating distances dict with geopy")
        distances = {}
        geolocator = Nominatim(user_agent="DSProject", timeout=20)
        mylist = ['Aleppo', 'Deir Ezzor', 'Homs', 'Daraa', 'Damascus Suburbs',
                  'Quneitra', 'Tartous', 'Damascus', 'Idlib', 'Sweida',
                  'Hama', 'Hasakeh', 'Lattakia', 'Raqqa']
        mohafaza_cities = {'north': "Zgharta-Ehden", 'south': "Sidon", 'beirut': "Beirut"}
        for place in mylist:
            distances[place] = {}
            for mohafaza, city in mohafaza_cities.items():
                print(place, mohafaza)
                location1 = geolocator.geocode(city)
                location2 = geolocator.geocode(place)
                latlng1 = (location1.latitude, location1.longitude)
                latlng2 = (location2.latitude, location2.longitude)
                dist_km = distance.distance(latlng1, latlng2).km
                dist_miles = distance.distance(latlng1, latlng2).miles
                distances[place][mohafaza] = {'km': dist_km, 'miles': dist_miles}
        # print("writing leb_syria_distances.p to file")
        # pickle.dump(distances, open("leb_syria_distances.p", "wb"))
        print('distances finished and they were:', distances)
        distance_km = []
        distance_miles = []
        for (i, row) in df.iterrows():
            if mohafaza_name is not None:
                mohafaza = mohafaza_name
            else:
                mohafaza = str(row['mohafaza'])
            event = str(row['place_of_death'])
            if event in ("Other Nationalities", "Unknown") or mohafaza not in ['north', 'south', 'beirut']:
                distance_km.append(np.NaN)
                distance_miles.append(np.NaN)
            else:
                dist_km = distances.get(event).get(mohafaza).get('km')
                dist_miles = distances.get(event).get(mohafaza).get('miles')
                distance_km.append(dist_km)
                distance_miles.append(dist_miles)
        df['distance_km'] = distance_km
        df['distance_miles'] = distance_miles
        return df


    @staticmethod
    def data_preprocessing(df):
        print("Preprocessing data")

        print("Delete village and patient id columns")
        df = df.drop(columns=['village', 'patid_pk'])

        print("Renaming columns")
        df = df.rename(index=str, columns={"cntid_pk": "center_id",
                                           "cntname": "center_name",
                                           "PServDate": "date",
                                           "Servname": "service_name"})
        print("new column names:")
        print(df.columns.tolist())

        print("convert center_id to numeric")
        # replace entry containing utf16-bom
        df = df.replace('\ufeff30202', 30202)
        df.center_id = pd.to_numeric(df.center_id)

        print("check center names match center_id (one to one mapping)")
        unique_pairs = len(df.groupby(['center_id', 'center_name']).count())
        unique_ids = len(df.center_id.unique())
        print(f"one to one mapping from ids to names: {unique_ids == unique_pairs}")

        print("change date column to date format")
        df.date = pd.to_datetime(df.date, format="%Y-%m-%d")
        print("remove entries outside of 2014 - 2016 range")
        df = df[df.date.dt.year >= 2014]
        df = df[df.date.dt.year <= 2016]
        
        print("Translate sex, nationality and service_name from arabic to english")
        df.sex.replace(sex_dict, inplace=True)
        df.mohafaza.replace(mohafaza_dict, inplace=True)
        df.nationality.replace(countries_dict, inplace=True)
        df.service_name.replace(services_dict, inplace=True)
        df.qada.replace(qada_dict, inplace=True)


        print("Replace unknown values with NaN")
        df.replace("not specified", np.NaN)

        return df

    @staticmethod
    def clean(df, sex, age, nationality, average_age):
        """
        Clean the data in a dataframe
        :param df: the dataframe to clean
        :param sex: bool, whether to clean the sex column
        :param age: bool, whether to clean the age column
        :param nationality: bool, whether to clean the nationality column
        :param average_age: bool, whether to assign an average age to nan values based on service
        :return: the clean dataframe
        """
        if age:
            print("set outlier age 117 to NaN")
            df.loc[df.age == 117, 'age'] = np.nan
            print("set age <= 0 to NaN")
            df.loc[df.age <= 0, 'age'] = np.nan
            print("set age > 90 to 90 (standard)")
            df.loc[df.age > 90, 'age'] = 90

        if sex:
            print("set child by age less than 16")
            df.loc[df.age < 16, "sex"] = "child"
            print("set child by service")
            df.loc[df.service_name == "Pediatrics", "sex"] = "child"
            print("set female by service")
            df.loc[(df.service_name == "Gynaecology") | (df.service_name == "Family Medicine"), "sex"] = "female"

        if nationality:
            print("set all other nationalities to other")
            df.loc[(df.nationality != "Lebanon") & (df.nationality != "Syria"), "nationality"] = "other"

        if average_age:
            print("replace unknown ages with the average for that service")
            for service in df.service_name.unique():
                mean_age = round(df.loc[df.service_name == service, 'age'].mean())
                df.loc[(df.age.isnull()) & (df.service_name == service), 'age'] = mean_age

        return df

    @staticmethod
    def dataframe_details(df):
        print(f"Dataframe has {df.shape[0]} entries across {df.shape[1]} columns")

    @staticmethod
    def add_external_data(df):
        print("mark which dates are public holidays")
        DataSet.add_public_holidays(df)
        print("add year, month and day columns")
        DataSet.add_time_columns(df)
        print("Adding sectarian data means per mohafaza")
        df = DataSet.add_sectarian_data(df)
        print("Adding socioeconomic data per mohafaza")
        df = DataSet.add_socioeconomic_data(df)
        print("Adding weather data")
        df = DataSet.add_weather_data(df)
        print("Adding syrian data")
        df = DataSet.add_syrian_data(df)
        print("Adding distance data")
        # df = DataSet.add_distances(df)
        return df

    @staticmethod
    def add_external_data_old(df):
        print("mark which dates are public holidays")
        DataSet.add_public_holidays(df)
        print("add year, month and day columns")
        DataSet.add_time_columns(df)
        print("Adding sectarian data means per mohafaza")
        df = DataSet.add_sectarian_data(df)
        print("Adding socioeconomic data per mohafaza")
        df = DataSet.add_socioeconomic_data(df)
        print("Adding weather data")
        df = DataSet.add_weather_data(df)
        print("Adding syrian data")
        df = DataSet.add_syrian_data_old(df)
        # print("Adding distance data")
        # df = DataSet.add_distances(df)
        return df

if __name__ == '__main__':
    ds = DataSet()
    ds.load_data_frame()
    # df = pd.read_csv('output/FaourProcessed.csv')

    # print(df.describe())
    # df = ds.clean(df, sex=True, age=True, nationality=True, average_age=True)
    # print(df.describe())

