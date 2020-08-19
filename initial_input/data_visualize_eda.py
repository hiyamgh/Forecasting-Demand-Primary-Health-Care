import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../..")
import glob
from scipy.signal import savgol_filter


class DataVisualizerEDA:

    def smooth_out(self, col):
        '''
        smoothing function that smooths out the curve
        :param col: the column to smooth out
        :return: smoothed out column
        '''
        if len(col) < 107:
            if len(col) % 2 == 0:
                wind_size = len(col) - 1
            else:
                wind_size = len(col) - 2
        else:
            wind_size = 107
        yhat = savgol_filter(col, wind_size, 8)
        return yhat

    def correlation_matrix(self, df, service_name, mohafaza, fig_name, output_folder):
        '''
        Creates an NxN correlation matrix that plots correlation between all variables
        :param df: data-subset: example: General Medicine_akkar.csv
        :param service_name: example: General Medicine
        :param mohafaza: example: akkar
        :param fig_name: name of the figure after its saved
        :param output_folder: the path to the output folder where it will be saved
        :return:
        '''
        corr = df.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        fig_title = service_name + ' in ' + mohafaza
        plt.title(fig_title)
        plt.savefig(output_folder + '/' + fig_name,  bbox_inches='tight')
        plt.close()

    def corr_matrix_highest_values(self, df, service_name, mohafaza, fig_name, output_folder):
        '''
        Creates a correlation matrix between 'demand' and 'resid' columns against all other columns
        :param df: data-subset: example: General Medicine_akkar.csv
        :param service_name: example: General Medicine
        :param mohafaza: example: akkar
        :param fig_name: name of the figure after its saved
        :param output_folder: the path to the output folder where it will be saved
        :return:
        '''
        corr = df.corr()
        plt.matshow(corr)
        corr = corr[['demand', 'resid']]
        # sort values by decreasing order of correlation values of demand
        corr = corr.sort_values(by=['demand', 'resid'], ascending=False)
        sns.heatmap(corr, annot=True) # annot=True shows the correlation value in each cell
        fig_title = service_name + ' in ' + mohafaza
        plt.title(fig_title)
        plt.savefig(output_folder + '/' + fig_name, bbox_inches='tight')
        plt.close()
        plt.show()

    def plot_demand_precipitation(self, df, service_name, mohafaza, fig_name, output_folder):
        '''
        creates two stacked line plots, one of 'demand', and another of 'Precipitation'
        :param df: data-subset: example: General Medicine_akkar.csv
        :param service_name: example: General Medicine
        :param mohafaza: example: akkar
        :param fig_name: name of the figure after its saved
        :param output_folder: the path to the output folder where it will be saved
        :return:
        '''

        # df = df.rename(index=str, columns={"Unnamed: 0": "date"})
        df['date'] = pd.to_datetime(df['date'])
        fig, axarr = plt.subplots(2, 1, figsize=(12, 8))
        smoothed = self.smooth_out(df['demand'])
        axarr[0].plot(df['date'], smoothed, label='demand')
        axarr[0].legend(loc='upper right')
        axarr[0].set_ylabel('demand')

        rank_smoothed = self.smooth_out(df['Precipitation'])
        # axarr[1].plot(df['date'], df[cols[1]], label=str(cols[1]))
        axarr[1].plot(df['date'], rank_smoothed, label='precipitation')
        axarr[1].legend(loc='upper right')
        axarr[1].set_ylabel('precipitation')

        for i in range(2):
            for tick in axarr[i].get_xticklabels():
                tick.set_rotation(45)

        title = service_name + ' in ' + mohafaza
        plt.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.savefig(output_folder + '/' + fig_name)
        plt.close()

    def line_plot(self, df, cols, service_name, mohafaza, fig_name, output_folder):
        fig, axarr = plt.subplots(len(cols), 1, figsize=(12, 8))
        for i in range(len(cols)):
            axarr[i].plot(df['date'], self.smooth_out(df[cols[i]]), label=str(cols[i]))
            axarr[i].legend()
        plt.suptitle('%s in %s' % (service_name, mohafaza))
        plt.savefig(output_folder + '/' + fig_name)
        plt.close()

    def triple_line_plot(self, df, cols, service_name, mohafaza, fig_name, output_folder):
        import peakutils
        # years = [2014, 2015, 2016]
        df = df.rename(index=str, columns={"Unnamed: 0": "date"})
        df['date'] = pd.to_datetime(df['date'])
        df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
        df['year'] = pd.to_datetime(df['date']).dt.year
        df_orig = df
        fig, axarr = plt.subplots(len(cols), 1, figsize=(12, 8))
        for i in range(len(cols)):
            index = peakutils.indexes(self.smooth_out(df[cols[i]]), thres=0.05, min_dist=1)
            axarr[i].plot(df['date'], self.smooth_out(df[cols[i]]), label=str(cols[i]))
            axarr[i].legend()
            # if i != len(cols) - 1:
            #     axarr[i].plot(df['date'][index], self.smooth_out(df[cols[i]])[index], marker='o', color='r', ls='')
        plt.suptitle('%s in %s' % (service_name, mohafaza))
        plt.savefig(output_folder + '/' + fig_name)
        plt.close()

    def triple_box_plot(self, df, cols, service_name, mohafaza, fig_name, output_folder):
        df = df.rename(index=str, columns={"Unnamed: 0": "date"})
        df['date'] = pd.to_datetime(df['date'])
        df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
        df['year'] = pd.to_datetime(df['date']).dt.year

        df = df[df['year'] != 2017]
        df.boxplot(column=cols, by=['year'], layout=(1, len(cols)))
        plt.suptitle('%s in %s' % (service_name, mohafaza))
        plt.savefig(output_folder + '/' + fig_name)
        plt.close()


if __name__ == '__main__':
    path = 'output/multivariate_datasubsets/*.csv'
    fnames = glob.glob(path)
    for f in fnames:
        df = pd.read_csv(f)
        df_name = f.split('\\')[1]
        service_name = df_name.split('_')[0]
        mohafaza = df_name.split('_')[1]
        mohafaza = mohafaza[:-4]
        fig_name = service_name + '_' + mohafaza + '.png'

        dv = DataVisualizerEDA()

        if not os.path.exists('output/EDA_plots/'):
            os.makedirs('output/EDA_plots/')

        folder_name = 'output/EDA_plots/demand_civ_rank_box/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        dv.triple_box_plot(df, ['demand', 'civilians_rank', 'distance'], service_name, mohafaza, fig_name, folder_name)

        folder_name = 'output/EDA_plots/demand_prec/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        dv.plot_demand_precipitation(df, service_name, mohafaza, fig_name, folder_name)

        folder_name = 'output/EDA_plots/demand_temp/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # dv.triple_line_plot(df, ['demand', 'MaxTemp', 'MinTemp'], service_name, mohafaza, fig_name, folder_name)
        dv.line_plot(df, ['demand', 'AverageTemp'], service_name, mohafaza, fig_name, folder_name)

        folder_name = 'output/EDA_plots/demand_wind/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # dv.triple_line_plot(df, ['demand', 'MaxWindSpeed', 'MinWindSpeed'], service_name, mohafaza, fig_name,
        #                     folder_name)
        dv.line_plot(df, ['demand', 'AverageWindSpeed'], service_name, mohafaza, fig_name, folder_name)

        folder_name = 'output/EDA_plots/demand_cause_rank_dist/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        dv.triple_line_plot(df, ['demand', 'cause_of_death_rank', 'distance'], service_name, mohafaza, fig_name,
                            folder_name)

        folder_name = 'output/EDA_plots/demand_actor_rank_dist/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        dv.triple_line_plot(df, ['demand', 'actor_rank', 'distance'], service_name, mohafaza, fig_name, folder_name)

        folder_name = 'output/EDA_plots/demand_civ_rank_dist/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        dv.triple_line_plot(df, ['demand', 'civilians_rank', 'distance'], service_name, mohafaza, fig_name, folder_name)


