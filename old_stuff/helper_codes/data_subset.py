from pandas import DataFrame
import pandas as pd
import numpy as np
import os.path
import sys
sys.path.append('..')
from old_stuff.helper_codes.data_set import DataSet


class DataSubset:
    def __init__(self, df):
        self.original_df = df

    def copy_df(self):
        return self.original_df.copy(deep=True)

    def agg_count(self, group_elements):
        """
        This method aggregates a dataframe by the count of all other rows
        :param group_elements: array of labels to group by
        :return:
        """
        df = self.copy_df()
        return DataFrame({'count': df.groupby(group_elements).size()}).reset_index()

    def agg_mean(self, group_elements):
        """
        This method aggregates a dataframe by the mean of a row
        :param group_elements: array of labels to group by
        :return:
        """
        df = self.copy_df()
        return df.groupby(group_elements).agg('mean')

    def categorize_demand(self, groupby=['date'], mohafaza='north', service='General Medicine'):
        """
        1- will aggregate the initial dataframe by the count of the date
        2- get the 1s1,2nd,3rd, and 4th quartile put into an array
        3- categorize the demand column ('here called: count') according to the count value
        :return: df with the categorization column added, called: 'demand_class'
        """
        df = self.agg_count(groupby)
        df = data_set.DataSet.add_external_data(df)
        if len(mohafaza):
            df = self.filter(df, 'mohafaza', [mohafaza])
        if len(service):
            df = self.filter(df, 'service_name', [service])
        desc = df.describe()
        min = desc.loc['min', 'count']
        fq = desc.loc['25%', 'count']
        sq = desc.loc['50%', 'count']
        tq = desc.loc['75%', 'count']
        max = desc.loc['max', 'count']
        nums = [min, fq, sq, tq, max]
        print(nums)
        classf = []
        for (i, row) in df.iterrows():
            dmnd = row['count']
            if dmnd <= fq:
                classf.append(0)
            elif dmnd >= tq:
                classf.append(2)
            else:
                classf.append(1)
        df['demand_class'] = classf
        return df


    def split_groups(self, df, group):
        """
        This method splits an aggregated dataframe into multiple dataframes
        :param df: dataframe
        :param group: array of labels to groupby
        :return:
        """
        groupped_df = df.groupby(group)
        return [groupped_df.get_group(group) for group in groupped_df.groups]

    def partition(self, df, column, num_partions):
        """
        Split a dataframe into partitions based on the number of unique values in a column
        :param df:  dataframe
        :param column: the column to partition on
        :param num_partions: the number of partitions to split the dataframe into
        :return: list of partitions
        """
        column_partition = np.array_split(df[column].unique(), num_partions)
        return [self.filter(df, column, values) for values in column_partition]

    @staticmethod
    def filter(df, column, values):
        """
        Returns a dataframe who's entries in a given column are only in a list of values
        :param df: dataframe
        :param column: column to apply filtering to
        :param values: list of values to include
        :return: dataframe who's entries in a given column are only in a list of values
        """
        return df[df[column].isin(values)]

    def generate_subsets(self, subset):
        """
        subsets dataframe
        :param subset: dict from column to list of values to use, put None instead of a list to use all values
        """
        df = self.copy_df()
        for column, values in subset.items():
            if values is not None:
                df = self.filter(df, column, values)
        return self.split_groups(df, list(subset.keys()))

    def save_subsets(self, subset, dir=""):
        """
        subsets dataframe and writes them to file as csv
        :param subset: dict from column to list of values to use, put None instead of a list to use all values
        """
        subset_list = self.generate_subsets(subset)

        for data_subset in subset_list:
            name = "ministry_of_health_subset_"
            subset_values = [data_subset[column].iloc[0] for column in subset.keys()]
            name += '_'.join(subset_values) + ".csv"
            name = name.replace(" ", "_")
            data_subset.to_csv(os.path.join(dir, name))


if __name__ == "__main__":
    data_set_holder = DataSet()
    # full_df = data_set_holder.copy_df()
    full_df = pd.read_csv('input/FaourProcessed.csv')
    full_df = data_set_holder.clean(full_df, sex=True, age=True, nationality=True, average_age=True)
    data_subsetter = DataSubset(full_df)
    data_subsetter.save_subsets({"nationality" : ["Lebanon", "Syria"],
                                 "mohafaza": ["north", "south", "beirut"],
                                 "service_name": ["General Medicine", "Pharmacy", "Pediatrics"]})