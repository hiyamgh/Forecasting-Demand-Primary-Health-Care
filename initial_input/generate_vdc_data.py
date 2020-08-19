import pandas as pd
import os
import matplotlib.pyplot as plt


def group_vdc_data(path):
    df_2014 = pd.read_csv(path + '2014_vdc_events.csv')
    df_2015 = pd.read_csv(path + '2015_vdc_events.csv')
    df_2016 = pd.read_csv(path + '2016_vdc_events.csv')

    # concatenate three datasets together
    df_grouped = pd.concat([df_2014, df_2015, df_2016])

    # transform the 'date_of_death' column to a categorcal one
    df_grouped['date_of_death'] = pd.to_datetime(df_grouped['date_of_death'])

    return df_grouped


class VDC:
    def __init__(self):
        ''' grouping the three data-sets: 2014, 2015, & 2016 into one '''
        if not os.path.isfile('input/vdc/vdc_grouped.csv'):
            df_grouped = group_vdc_data('input/vdc/')
            df_grouped.to_csv('input/vdc/vdc_grouped.csv', index=False)
            self.df = df_grouped
        else:
            self.df = pd.read_csv('input/vdc/vdc_grouped.csv')

    def casualties(self, output_folder):
        df = self.df

        df = df[['date_of_death', 'nb_civilians']]

        df = df.rename(columns={'date_of_death': 'date'})

        df = df.groupby('date').agg({'nb_civilians': 'sum'})

        df.index = pd.to_datetime(df.index)

        df = df.resample('W-TUE').sum()

        quantile_list = [0, .25, .5, .75, 1.]
        quantiles = df['nb_civilians'].quantile(quantile_list)

        df['nb_civilians'].hist(bins=30, color='#A9C5D3',
                                edgecolor='black', grid=False)
        plt.xlabel('Number of civilians')
        plt.ylabel('Frequency')
        plt.title('Histogram showing the distribution of the number of civilians')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + 'civilians_distribution_before_bins.png')
        plt.close()

        fig, ax = plt.subplots()
        df['nb_civilians'].hist(bins=30, color='#A9C5D3',
                                edgecolor='black', grid=False)
        for quantile in quantiles:
            qvl = plt.axvline(quantile, color='r')
        ax.legend([qvl], ['Quantiles'], fontsize=10)
        plt.xlabel('Number of civilians')
        plt.ylabel('Frequency')
        plt.title('Histogram of distribution of civilians with bins annotated')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + 'civilians_distribution_after_bins.png')
        plt.close()

        quantile_labels = [1, 2, 3, 4]

        # quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
        df['civ_quantile_range'] = pd.qcut(df['nb_civilians'], q=quantile_list)
        df['civ_quantile_label'] = pd.qcut(df['nb_civilians'],q=quantile_list,
            labels=quantile_labels)

        print(df.head(10))
        df = df[['nb_civilians', 'civ_quantile_label']]
        dest = 'output/vdc/'
        df.to_csv(dest + 'civilians_rank.csv')


def get_casualties_by_month_year(output_folder):
    import seaborn as sns
    sns.set(style="whitegrid")
    # read the data
    df = pd.read_csv('output/vdc/civilians_rank.csv')

    # get the month-year of each week
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
    df['year'] = pd.to_datetime(df['date']).dt.year

    # we don't need 207 because our data is 2014-->2016
    df = df[df['year'] != 2017]

    for time in ['month_year', 'year']:
        # df.boxplot(column='nb_civilians', by=[time], layout=(1, 1))
        sns.boxplot(x=time, y='nb_civilians', data=df)
        fig = plt.gcf()
        if time == 'month_year':
            fig.set_size_inches(16, 10)
        plt.xticks(rotation=45)
        if time == 'month_year':
            plt.xlabel('Month-Year')
        else:
            plt.xlabel('Year')
        plt.ylabel('Number of civilians')
        plt.title('%s Distribution of Civilians' % ('Monthly' if time == 'month_year' else 'Yearly'))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + 'civilians_%sly.png' % time)
        plt.close()


def eda_syrian_cities(output_folder):
    df = pd.read_csv('input/vdc/vdc_grouped.csv')

    # replace 'Damascus Suburbs' with Damascus
    df.loc[df.place_of_death == 'Damascus Suburbs', 'place_of_death'] = 'Damascus'

    # group by sum of civilians per place of death
    df = df.groupby(['place_of_death']).agg({'nb_civilians': 'sum'})

    # get the labels and the number of civilians
    labels = list(df.index)
    civ = list(df['nb_civilians'])

    # pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(civ, labels=labels, autopct='%5.1f%%', shadow=True, startangle=90, pctdistance=0.85)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1.set_size_inches(15, 10)

    plt.suptitle('Number of Civilians in Each Syrian City')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + 'place_of_death_pie.png')
    plt.close()


def get_distances_from_syrian_cities(output_folder):
    distances = pd.read_csv('input/distance_dictionary/distances.csv')
    distances = distances.set_index('Unnamed: 0')
    mydicts = {}

    mohafazas = {'akkar': 'Akkar', 'bikaa': 'Beqaa', 'north': 'Tripoli'}
    for i, col in enumerate(['akkar', 'bikaa', 'north']):
        mydicts[col] = {}
        curr = dict((key, distances.loc[key][col]) for key in list(distances.index))
        curr = dict(sorted(curr.items(), key=lambda kv: kv[1]))
        mydicts[col] = curr
    print(mydicts)

    fig, ax = plt.subplots(3, 1)
    ll = ['akkar', 'bikaa', 'north']
    for i in range(3):
        m = ll[i]
        cities = list(mydicts[m].keys())
        dists = list(mydicts[m].values())
        ax[i].bar(cities, dists)
        ax[i].tick_params(axis='x', which='major', labelsize=8)
        ax[i].set_title('%s' % mohafazas[m])
        ax[i].set_ylabel('distance in Km')
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
    fig.tight_layout()
    fig.set_size_inches(15, 9)
    plt.suptitle('Distances from the Syrian cities to the Governorates')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + 'distances_mohafazas.png')
    plt.close()


if __name__ == '__main__':
    vdc = VDC()
    vdc.casualties(output_folder='eda_plots/vdc_plots/')
    eda_syrian_cities('eda_plots/vdc_plots/')
    # get_distances_from_syrian_cities('eda_plots/vdc_plots/')
    get_casualties_by_month_year('eda_plots/vdc_plots/')



