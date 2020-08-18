import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from old_stuff.helper_codes.data_set import DataSet
from old_stuff.helper_codes.data_subset import DataSubset
from pandas.plotting import autocorrelation_plot
from numpy import sqrt
from numpy import log
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.stats.diagnostic as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import glob
import os
import sys
sys.path.append("../..")
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class TimeSeries:

    '''
    Time Series class specific to the data subsets of this project (MoPH)
    '''

    def __init__(self, data_frame=None, downsample=None, rolling_average_window=None, filter=None):
        """
        Initialize the time series class
        additional data manipulation will not be done if params are left as None (or False)
        :param downsample: string, how much to downsample by (e.g. 'W-TUE': weekly, '2W-TUE': biweekly, 'M': month)
        :param rolling_average_window: int, number of days to average over if applying rolling average to count
        :param filter: dict, which columns/values to filter before aggregating by date format is {column: [values]}
        """
        data_set_holder = DataSet()
        # added this 'if statement' here
        if data_frame is None:
            # data_set_holder = data_set.DataSet()
            self.df = data_set_holder.copy_df()
        else:
            self.df = data_frame
            self.df['date'] = pd.to_datetime(self.df['date'])
        if filter:
            for column, value in filter.items():
                self.df = DataSubset.filter(self.df, column, value)
        subsetter = DataSubset(self.df)
        self.series_df = subsetter.agg_count(['date'])
        #self.find_missing()
        data_set_holder.add_time_columns(self.series_df)
        self.series = self.series_df.set_index('date')['count']
        if downsample:
            if downsample == '2W-TUE':
                self.series = self.series.resample(downsample).sum()
            else:
                self.series = self.series.resample(downsample).sum()
        # smooth data with a rolling average
        if rolling_average_window:
            self.series = self.rolling_avg(rolling_average_window)
            # replace nan values at start with copies of the first value
            # this is required to make some of the functions work
            for i in range(rolling_average_window - 1):
                self.series[i] = self.series[rolling_average_window]
        self.downsample = downsample

    def rolling_avg(self, window_size):
        """
        create a rolling average of the time series
        :param window_size: int the size of the rolling average window
        :return: Series, the time series with average smoothing
        """
        rolling = self.series.rolling(window=window_size)
        return rolling.mean()


class TimeSeriesPlotter:

    '''
    This class can be used for generating time series - related plots. We can use this class with any
    data frame, not just the ones we are dealing with in this project
    '''

    def __init__(self, df, target_variable, service_name=None, mohafaza=None):
        '''
        :param df: dataframe being used
        :param target_variable: name of the target variable column
        :param service_name: name of the service of this dataframe. By default it is None.
        :param mohafaza: name of the Governorate (mohafaza) of this dataframe. By default is is None.
        '''
        self.df = df
        self.target_variable = target_variable
        self.service_name = service_name
        self.mohafaza = mohafaza

    def create_folder(self, folder_path):
        '''
        creates a folder in the specified path
        :param folder_path: path to the folder
        :return:
        '''
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def create_title_image_name(self, image_title, image_name):
        '''
        this function is responsible for creating a title of the image being created, and the name
        of the image created when it is saved
        :param image_title: title that will appear in the plot
        :param image_name: name of the image that will be saved. Example: if `line_plot` is
        passed, it will be saved ad `line_plot.png`
        :return: the title and the name of the image
        '''
        if self.service_name is not None and self.mohafaza is not None:
            image_title = '%s of %s for %s in %s' % (image_title, self.target_variable, self.service_name, self.mohafaza)
            image_name = '%s_%s_%s.png' % (self.service_name, self.mohafaza, image_name)
        else:
            image_title = '%s of %s' % (image_title, self.target_variable)
            image_name = '%s.png' % image_name
        return image_title, image_name

    # function for generating a line plot of a specified column in a specified dataframe
    def generate_lineplot(self, output_folder):
        '''
        this function creates a line plot of the target variable
        :param output_folder: path to the folder where the line plot will be saved
        :return saves the line plot image in the specified folder
        '''
        self.df[self.target_variable].plot()
        image_title, image_name = self.create_title_image_name('Line plot', 'line_plot')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()

    # function for generating a histogram of a specified column in a specified dataframe
    def generate_hist(self, output_folder):
        '''
        this function creates a histogram of the target variable
        :param output_folder: path to the folder where the histogram will be saved
        :return saves the line plot image in the specified folder
        '''
        self.df[self.target_variable].hist()
        image_title, image_name = self.create_title_image_name('Histogram', 'histogram')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()

    # function for generating a line plot of the square root of a specified column in a specified dataframe
    def generate_sqrt_transform(self, output_folder):
        '''
        this function generates a line plot of the square root transform applied to the target variable passed
        as a parameter of this class.
        :param output_folder: path to the folder where the line plot of the square root transform will be saved
        :return: the square root series of the target variable
        '''
        series_sqrt = sqrt(self.df[self.target_variable])
        plt.plot(series_sqrt)
        image_title, image_name = self.create_title_image_name('Square Root Transform', 'sqrt_transform')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()
        return series_sqrt

    # function for generating a line plot of the log transform of a specified column in a specified dataframe
    def generate_log_transform(self, output_folder):
        '''
        this function generates a line plot of the log transform applied to the target variable passed
        as a parameter of this class.
        :param output_folder: path to the folder where the line plot of the log transform will be saved
        :return: the log transform series of the target variable
        '''
        series_log = log(self.df[self.target_variable])
        series_log = series_log[series_log != -np.inf]
        image_title, image_name = self.create_title_image_name('Log Transform', 'log_transform')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()
        return series_log

    # function for generating a line plot of the box cox transform of a specified column in a specified dataframe
    def generate_boxcox(self, output_folder):
        '''
        this function generates a line plot of the box cox transform applied to the target variable passed
        as a parameter of this class.
        :param output_folder: path to the folder where the line plot of the box cox transform will be saved
        :return: the box cox transform series of the target variable
        '''
        series_bc = self.df[self.target_variable][self.df[self.target_variable] != 0]
        series_bc, lam = boxcox(series_bc)
        print('Lambda: %f' % lam)
        image_title, image_name = self.create_title_image_name('Box Cox Transform', 'box_cox_transform')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()
        return series_bc

    # function for generating a density plot of the target variable of the dataframe being passed
    def density_plot(self, output_folder):
        '''
        function that generates a density plot of the target variable of the dataframe being passed
        :param output_folder: path to the folder where the density plot of the target variable will be saved
        :return: saves the density plot as an image in the specified output folder
        '''
        self.df[self.target_variable].plot(kind='kde')
        image_title, image_name = self.create_title_image_name('Density Plot', 'density_plot')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()

    # function that performs Ljung Box Test for detecting white noise on the target variable of the dataframe
    # being passed to this class
    def ljung_box_test(self, output_folder, df_name):
        '''
        function that applies L-jung box test for detecting white noise in the target variable of the
        dataframe being passed as a parameter of this class
        :param output_folder: path to the output folder where the dataframe that contains
        the columns returned by the Ljung-Box test will be saved
        :param df_name: name that is associated to the dataframe that will be created
        :return: the dataframe created
        '''

        if self.service_name is not None and self.mohafaza is not None:
            print("testing for %s in %s" % (self.service_name, self.mohafaza))
        arr = sm.acorr_ljungbox(self.df[self.target_variable], boxpierce=True)
        df = pd.DataFrame({'lb': arr[0], 'p-values': arr[1], 'bpvalue': arr[2], 'bpp-values': arr[3]})
        df.index.name = 'lag_nb'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df.to_csv(output_folder + df_name + '.csv')
        if len(df[df['p-values'] <= 0.05]) == len(df):
            print('all p-values for ljung box <= 0.05')
        if len(df[df['bpp-values'] <= 0.05]) == len(df):
            print('all p-values for box pierce <= 0.05')
        print('-----------------------------------------------')
        return df

    # function that performs augmented dickey-fuller test on the target variable of the dataframe
    # being passed to this class
    def dickey_fuller(self, reg='ct'):
        '''
        function that applies Augmented Dickey Fuller Test for stationarity checking of the target variable
        of the dataframe being passed
        :param reg: the regression value used in the adfuller function of statsmodel. By default, it is
        `ct`. The possible values are: `c`, `ct`, `ctt`, `nc`. references: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
        :return: prints out the p-value as well as the critical values
        '''
        if self.service_name is not None and self.mohafaza is not None:
            print("testing for %s in %s" % (self.service_name, self.mohafaza))
        X = self.df[self.target_variable].values
        result = adfuller(X, regression=reg)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Lags used: %d' % result[2])
        print('Critical Values:')
        for key, value in result[4].items():
            print('%s: %.3f' % (key, value))
        print('-----------------------------------------------')

    # function that generates ACF plot of the target variable of the dataframe passed to this class
    def generate_acf(self, output_folder):
        '''
        function that generates and ACF plot of the target variable of the dataframe being passed to this class
        :param output_folder: path to the folder where the acf plot of the target variable will be saved
        :return: saves the acf plot as an image in the specified output folder
        '''
        plot_acf(self.df[self.target_variable])
        image_title, image_name = self.create_title_image_name('ACF plot', 'acf_plot')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()

    # function that generates PACF plot of the target variable of the dataframe passed to this class
    def generate_pacf(self, output_folder):
        '''
        function that generates and PACF plot of the target variable of the dataframe being passed to this class
        :param output_folder: path to the folder where the pacf plot of the target variable will be saved
        :return: saves the pacf plot as an image in the specified output folder
        '''
        plot_pacf(self.df[self.target_variable])
        image_title, image_name = self.create_title_image_name('PACF Plot', 'pacf_plot')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()

    # function that generates Autocorrelation plot of the target variable of the dataframe passed to this class
    def generate_autocorrelation(self, output_folder):
        '''
        function that generates and acuto-correlation plot of the target variable of the dataframe being passed to this class
        :param output_folder: path to the folder where the auto-correlation plot of the target variable will be saved
        :return: saves the auto-correlation plot as an image in the specified output folder
        '''
        autocorrelation_plot(self.df[self.target_variable])
        image_title, image_name = self.create_title_image_name('Autocorrelation Plot', 'autocorrelation_plot')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()

    def persistence_model(self, output_folder):
        df = pd.DataFrame(self.df[self.target_variable])
        df = add_lags(df, self.target_variable, 1, 't-')
        df.columns = ['t+1', 't-1']

        # add the data frame to persistence model input folder
        destination = '../input/persistence_input/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        df.to_csv(destination + '%s_%s.csv' % (self.service_name, self.mohafaza))

        X = df.values

        train_size = int(len(X) * 0.80)
        train, test = X[1:train_size], X[train_size:]
        train_X, train_y = train[:, 1], train[:, 0]
        test_X, test_y = test[:, 1], test[:, 0]

        # evaluate persistence model
        predictions = []
        for x in test_X:
            predictions.append(x)

        r2_Score = r2_score(test_y, predictions)
        rmse_score = np.sqrt(mean_squared_error(test_y, predictions))
        mse_score = mean_squared_error(test_y, predictions)
        mae_score = mean_absolute_error(test_y, predictions)
        mape_score = mean_absolute_percentage_error(test_y, predictions)

        print('Persistence model Scores:\nR^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
              (r2_Score, rmse_score, mse_score, mae_score, mape_score))

        # plot the persistence model, actual vs. predicted
        plt.plot(test_y, label='actual')
        plt.plot(predictions, label='predicted')
        plt.legend()
        image_title, image_name = self.create_title_image_name('Persistence model', 'persistence_plot')
        plt.suptitle(image_title)
        self.create_folder(output_folder)
        plt.savefig(output_folder + image_name)
        plt.close()


def mean_absolute_percentage_error(y_true, y_pred):
    '''
    Function to compute the mean absolute percentage error (MAPE) between an actual and
    predicted vectors
    :param y_true: the actual values
    :param y_pred: the predicted values
    :return: MAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# function that extracts the time series components (trend, seasonality, and residual) from the target variable of
# the dataframe passed to this function
def get_timeseries_components(df, target_variable, model='additive', freq=None):
    '''
    function that extracts: trend, seasonality, and residual from the target variable of the dataframe
    passed to this function
    :param freq: the length of the seasonal cycle in the data
    :param model: whether the seasonal decomposition is 'additive' or 'multiplicative'
    :return: trend, seasonality, and the residual components of the target variable
    '''
    if freq is None:
        result = seasonal_decompose(df[target_variable], model=model, extrapolate_trend='freq')
    else:
        result = seasonal_decompose(df[target_variable], model=model, freq=freq, extrapolate_trend='freq')
    trend = result.trend
    season = result.seasonal
    # resid = result.resid

    df['%s_trend' % target_variable] = trend
    df['%s_seasonality' % target_variable] = season
    # df['%s_resid' % target_variable] = resid
    return df


# function that creates certain number of lags from the target variable of the dataframe passed to this function
def add_lags(df, target_variable, nb_lags, col_prefix, col_suffix=None):
    '''
    function that creates a specified number of lag columns of the target variable of the
    dataframe passed.
    :param df: dataframe
    :param nb_lags: number of lags that will be added as columns. Example: if nb_lags = 3, the first
    three lags will be created
    :param target_variable: the target variable in the dataframe passed. Lags will be created from thiscolumn
    :param col_prefix: prefix of the lag column that will be created. Example, if 't-' is passed as a prefix
    and the number of lags is 3, then the column names will be: `t-1`, `t-2`, and `t-3` (one column for each lag from 1 to nb_lags)
    :param col_suffix: suffix of the lag column that will be created. Example, if '{t-' is the prefix and
    '}' is the suffix, and the number of lags is 3, then the column names will be `{t-1}`, `{t-2}`,
    and `{t-3}` (one column for each lag from 1 to nb_lags)
    :return: the lag columns created
    '''
    for i in range(1, nb_lags + 1):
        if col_suffix is None:
            lag_label = col_prefix + str(i)
            df[lag_label] = df[target_variable].shift(i)
        else:
            lag_label = col_prefix + str(i) + col_suffix
            df[lag_label] = df[target_variable].shift(i)
    return df


# function that applies difference transform to the target variable of the dataframe passed to this function
def difference_series(df, target_variable, interval=1):
    '''
    function that applies a difference transform to the target variable of the dataframe passed
    :param df: dataframe
    :param target_variable: the column that we will apply difference transform to.
    :param interval: by how much should we difference. By default, it is 1. If 1, a difference
    transform by an interval of 1 will be applied
    :return: the differenced series of the target variable.
    '''
    diff_demand = list()
    for i in range(interval, len(df)):
        value = df[target_variable][i] - df[target_variable][i - interval]
        diff_demand.append(value)

    df = df.drop(df.index[-interval])
    df[target_variable] = diff_demand
    return df


def generate_plots():
    # fnames = glob.glob('../old_stuff/output/Faour_datasubsets/*.csv')
    fnames = glob.glob('../old_stuff/output/multivariate_datasubsets/*.csv')
    output_folder = '../old_stuff/output/temporal_structure_plots/'
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']

    for f in fnames:
        for service in services:
            for mohafaza in mohafazas:
                if service in f and mohafaza in f:
                    df = pd.read_csv(f)

                    print('DATAFRAME: %s in %s' % (service, mohafaza))
                    tp = TimeSeriesPlotter(df, 'demand', service_name=service, mohafaza=mohafaza)
                    print('%s in %s' % (service, mohafaza))
                    tp.generate_lineplot(output_folder + 'line_plot/')
                    tp.generate_hist(output_folder + 'histogram/')
                    tp.density_plot(output_folder + '/density_plot/')
                    tp.ljung_box_test(output_folder + 'ljung_box_results/', '%s_%s' % (service, mohafaza))
                    tp.dickey_fuller()
                    tp.generate_acf(output_folder + 'acf/')
                    tp.generate_pacf(output_folder + 'pacf/')
                    tp.generate_autocorrelation(output_folder + 'autocorrelation/')
                    tp.persistence_model('../output/persistence_output/plots/')


if __name__ == '__main__':
    generate_plots()

