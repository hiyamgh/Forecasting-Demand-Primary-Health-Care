import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from matplotlib.dates import MonthLocator
from sklearn.model_selection import train_test_split

import data_set
from data_subset import DataSubset


class Visualizer:

    def box_plot(self, df, by, title=''):
        """
        Drawes a box plot from panda's dataframe
        :argument df frame
        :argument by group by element
        """
        df.boxplot(by=by, figsize=(10, 10))
        plt.xticks(rotation=90)
        plt.title(title)
        plt.show()

    def lines_plot(self, frames, groupper, x='date', y='count', freq='1M'):
        """
        draws a multi line plot from an array of panda's data frames
        :argument frames array of data frames
        :argument groupper group by element
        :argument x  plot xaxis
        :argument y plot yaxis
        :argument freq data frequency for xaxis ticks
        """
        fig, ax = plt.subplots()
        for frame in frames:
            mohafaza = frame[groupper].iloc[0]  # gets the name of the mohafaza from the first row
            frame.set_index(x, inplace=True)
            frame.index = pd.to_datetime(frame.index)
            frame = frame.groupby(pd.Grouper(freq=freq)).sum().reset_index()
            fig.set_figheight(10)
            fig.set_figwidth(10)
            ax.plot(frame[x], frame[y], label=mohafaza)
        plt.xticks(rotation=90)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
        plt.show()

    def pie_plot(self, df, x, y, title=''):
        """
        draws a pie plot from pandas daraframe
        :param df: panda's dataframe
        :param x: elements sizes index
        :param y: element labels index
        :param title: pie plot title
        :return:
        """
        sizes = df[x]
        labels = df[y]
        explode = 0.01
        explode = (explode,) * len(labels)
        pie = plt.pie(sizes, labels=labels,
                      autopct='%1.1f%%', shadow=True, startangle=140, explode=explode)
        plt.title(title)
        plt.legend(pie[0], labels, loc='best')
        plt.show()

    def time_bar_plot(self, df, groupper='', index='date', freq='3M', title=''):
        df.set_index(index, inplace=True)
        df.index = pd.to_datetime(df.index)
        if groupper != '':
            df = df.groupby([groupper, pd.Grouper(freq=freq)]).sum()
        else:
            df = df.groupby(pd.Grouper(freq=freq)).sum()
        df.plot(kind='bar', stacked=True, figsize=(10, 10))
        plt.title(title)
        plt.show()

    def bar_plot(self, df, x, y, title=''):
        df.plot(x=x, y=y, kind='bar', figsize=(10, 10))
        plt.title(title)
        plt.show()


class Plotter:
    def __init__(self):
        data_set_holder = data_set.DataSet()
        df = data_set_holder.copy_df()
        df = data_set_holder.clean(df, sex=True, age=True, nationality=True, average_age=True)
        self.data_subsetter = DataSubset(df)
        self.visualizer = Visualizer()

    def service(self):
        """
        bar plot showing the total demand on each service
        """
        service_name_dataframe = self.data_subsetter.agg_count(['service_name'])
        self.visualizer.bar_plot(service_name_dataframe, x='service_name', y='count')

        """
        box plots to show the demand on each service per month
        """

        service_name_df = self.data_subsetter.agg_count(['service_name', 'date'])
        services_box_array = self.data_subsetter.partition(service_name_df, 'service_name', 4)
        for partitioned_df in services_box_array:
            self.visualizer.box_plot(partitioned_df, 'service_name', '')

        """
        multiple lines plot that shows the demand per service in 1month frequency
        """
        for partitioned_df in services_box_array:
            services_split = self.data_subsetter.split_groups(partitioned_df, ['service_name'])
            self.visualizer.lines_plot(services_split, 'service_name')

        """
        multiple lines plot that shows the demand per service in  6 months frequency
        """
        for partitioned_df in services_box_array:
            services_split = self.data_subsetter.split_groups(partitioned_df, ['service_name'])
            self.visualizer.lines_plot(services_split, 'service_name', freq="6M")

        """
        bar plot showing the demand per each service based on 9 months frequency
        """
        services_lines_plot_split = self.data_subsetter.partition(service_name_df, 'service_name', 8)
        for df in services_lines_plot_split:
            self.visualizer.time_bar_plot(df, groupper='service_name', freq='9M')

    def nationality(self):

        """
        box plot to show the distrbution of nationality, each point refers to a day
        """
        nationality_df = self.data_subsetter.agg_count(['nationality', 'date'])
        self.visualizer.box_plot(nationality_df, 'nationality')

        """
        box plots that shows the demand in each mohafaza per nationality, each point refers to a day
        """
        mohafaza_nationality_df = self.data_subsetter.agg_count(['mohafaza', 'nationality', 'date'])
        mohafaza_box_array = self.data_subsetter.partition(mohafaza_nationality_df, 'mohafaza', 2)
        for paritioned_df in mohafaza_box_array:
            self.visualizer.box_plot(paritioned_df, ['mohafaza', 'nationality'])

        """
        pie plot that shows the total percentages of nationality
        """
        nationality_df = self.data_subsetter.agg_count(['nationality'])
        self.visualizer.pie_plot(nationality_df, x='count', y='nationality', title='nationality pie chart')

        """
        multiple lines plot that shows the demand per nationality
        """
        line_nationality_plot = self.data_subsetter.agg_count(['nationality', 'date'])
        nationality_df = self.data_subsetter.split_groups(line_nationality_plot, 'nationality')
        self.visualizer.lines_plot(nationality_df, 'nationality')

        """
         bar plot showing the demand per nationality based on 6 months frequency
        """
        self.visualizer.time_bar_plot(line_nationality_plot, groupper='nationality', freq='6M')

    def mohafaza(self):

        """
        box plot that shows the demand in each mohafaza, each point refers to a day
        """
        mohafaza_df = self.data_subsetter.agg_count(['mohafaza', 'date'])
        self.visualizer.box_plot(mohafaza_df, 'mohafaza')

        """
            multiple lines plot that shows the demand per mohafaza
            """
        line_mohafaza_df = self.data_subsetter.agg_count(['mohafaza', 'date'])
        mohafaza_df = self.data_subsetter.split_groups(line_mohafaza_df, 'mohafaza')
        self.visualizer.lines_plot(mohafaza_df, 'mohafaza')

    def sex(self):
        """
        box plots that shows the demand in each mohafaza per sex, each point refers to a day
        """
        mohafaza_sex = self.data_subsetter.agg_count(['mohafaza', 'sex', 'date'])
        mohafaza_sex_array = self.data_subsetter.partition(mohafaza_sex, 'mohafaza', 4)
        for paritioned_df in mohafaza_sex_array:
            self.visualizer.box_plot(paritioned_df, ['mohafaza', 'sex'])

        """
        box plots that shows the demand in each nationality per sex, each point refers to a day
        """
        nationality_sex_df = self.data_subsetter.agg_count(['nationality', 'sex', 'date'])
        self.visualizer.box_plot(nationality_sex_df, ['nationality', 'sex'])

        """
        pie plot that shows the total percentages of sex
        """
        sex_df = self.data_subsetter.agg_count(['sex'])
        self.visualizer.pie_plot(sex_df, x='count', y='sex')
        self.visualizer.bar_plot(sex_df, x='sex', y='count')

        """
        pie plot that shows the total percentages of sex per mohafaza
        """
        sex_mohafaza_df = self.data_subsetter.agg_count(['mohafaza', 'sex'])
        sex_dataframe = self.data_subsetter.split_groups(sex_mohafaza_df, 'mohafaza')
        for frame in sex_dataframe:
            self.visualizer.pie_plot(frame, x='count', y='sex', title="Sex pie chart in " + frame['mohafaza'].iloc[0])

        """
        pie plot that shows the total percentages of sex per mohafaza per nationality
        """
        nationality_mohafaza_df = self.data_subsetter.agg_count(['mohafaza', 'nationality'])
        nationality_dataframe = self.data_subsetter.split_groups(nationality_mohafaza_df, 'mohafaza')
        for frame in nationality_dataframe:
            self.visualizer.pie_plot(frame, title="Nationality pie chart in " + frame['mohafaza'].iloc[0], x='count',
                                     y='nationality')

        """
        multiple lines plot that shows the demand per sex
        """
        line_sex_df = self.data_subsetter.agg_count(['sex', 'date'])
        sex_df = self.data_subsetter.split_groups(line_sex_df, 'sex')
        self.visualizer.lines_plot(sex_df, 'sex')

    def public_holidays(self):

        """
        boxplot showing the total demand dependant on whether it is a public holiday
        """
        holiday_df = self.data_subsetter.agg_count(['holiday', 'date'])
        self.visualizer.box_plot(holiday_df, 'holiday', 'distribution of demand on public holidays')

        """
        boxplot showing the total demand dependant on whether it is a public holiday per mohafaza
        """
        holiday_df = self.data_subsetter.agg_count(['holiday', 'mohafaza', 'date'])
        holiday_mohafaza_dfs = self.data_subsetter.split_groups(holiday_df, 'mohafaza')
        for frame in holiday_mohafaza_dfs:
            self.visualizer.box_plot(frame, 'holiday', f"{frame['mohafaza'].iloc[0]} demand by day of the week")

    def week_days(self):
        """
        boxplot showing the total demand based on the day of the week
        """
        weekday_df = self.data_subsetter.agg_count(['dayofweek', 'date'])
        self.visualizer.box_plot(weekday_df, 'dayofweek', 'disitrbution of demand by day of the week')

        """
        boxplot showing the total demand based on the day of the week by service
        """
        # get the top 5 services
        service_name_dataframe = self.data_subsetter.agg_count(['service_name'])
        top_services = service_name_dataframe.nlargest(5, 'count')['service_name'].tolist()

        weekday_df = self.data_subsetter.agg_count(['dayofweek', 'service_name', 'date'])
        weekday_df = self.data_subsetter.filter(weekday_df, "service_name", top_services)
        service_dfs = self.data_subsetter.split_groups(weekday_df, 'service_name')
        for frame in service_dfs:
            self.visualizer.box_plot(frame, 'dayofweek', f"{frame['service_name'].iloc[0]} demand by day of the week")

    @staticmethod
    def weather():
        weather_df = pd.read_csv("Weather.csv", parse_dates=["Date"])
        weather_series = weather_df.set_index('Date')['MinTemp']
        weather_series.hist()
        plt.show()
        min_weeks = weather_series.resample("W-MON").min()
        min_weeks.plot()
        plt.ylabel("Temp")
        plt.title("Minimum Temperature in Beirut")
        plt.show()

        weather_series = weather_df.set_index('Date')['MaxTemp']
        max_weeks = weather_series.resample("W-MON").max()
        max_weeks.plot()
        plt.ylabel("Temp")
        plt.title("Maximum Temperature in Beirut")
        plt.show()


plotter = Plotter()
plotter.service()
plotter.nationality()
plotter.mohafaza()
plotter.sex()
plotter.public_holidays()
plotter.week_days()
plotter.weather()
