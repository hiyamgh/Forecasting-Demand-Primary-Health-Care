import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


def create_lineplot(df, cols, output_folder, fig_name):
    fig, axarr = plt.subplots(len(cols), 1, figsize=(12, 8))
    for i in range(len(cols)):
        axarr[i].plot(pd.to_datetime(df['date']), df[cols[i]], label=str(cols[i]))
        axarr[i].legend()
        axarr[i].set_ylabel(cols[i])
        axarr[i].tick_params(axis='x', rotation=45)
    fig.tight_layout()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + '/' + fig_name)
    plt.close()


def corr_matrix(df, service_name, mohafaza, output_folder):
    corr = df.corr()
    sns.heatmap(corr,  center=0, linewidths=1, annot=True, fmt=".2f")
    plt.xticks(fontsize=8, rotation=30)
    plt.yticks(fontsize=8, rotation=30)
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.title('Correlation Matrix for %s in %s' % (service_name, mohafaza))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + '%s_%s.png' % (service_name, mohafaza))
    plt.close()


def boxplot_collated(df, target_variable, output_folder, fig_name, by_year=False, all=False):
    if not all:
        df['date'] = pd.to_datetime(df['date'])
        df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
        df['year'] = pd.to_datetime(df['date']).dt.year

        # we don't need 2017 because our data is 2014-->2016
        df = df[df['year'] != 2017]

        if by_year:
            ax = sns.boxplot(x='year', y=target_variable, data=df)
        else:
            ax = sns.boxplot(x='month_year', y=target_variable, data=df)
        if not by_year:
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            fig.tight_layout()
        plt.xticks(rotation=45)
        plt.title('Boxplot %s Distribution of %s' % ('Monthly' if not by_year else 'Yearly', target_variable))

        ax.set(xlabel='Month - Year' if not by_year else 'Year',
               ylabel='%s' % target_variable)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.savefig(output_folder + '%s.png' % fig_name)
        plt.close()
    else:
        sns.boxplot(x=df[target_variable])
        plt.title('BoxPlot of %s' % target_variable)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.savefig(output_folder + '%s.png' % fig_name)
        plt.close()


def demand_boxplot(df, service_name, mohafaza, output_folder):
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
    df['year'] = pd.to_datetime(df['date']).dt.year

    # we don't need 2017 because our data is 2014-->2016
    df = df[df['year'] != 2017]

    sns.boxplot(x='month_year', y='demand', data=df)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.title('Boxplot Monthly Distribution of Demand for %s in %s' % (service, mohafaza))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(output_folder + '%s_%s.png' % (service_name, mohafaza))
    plt.close()


def corr_civ_demand(output_folder):
    ''' get the data subset most correlated with civilians rank '''
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']
    pearson_corr = {}
    for service in services:
        for mohafaza in mohafazas:
            df = pd.read_csv('output/multivariate_datasubsets/%s_%s.csv' % (service, mohafaza))
            civ_rank = df['civilians_rank']
            demand = df['demand']
            pearson_corr['%s_%s' % (service, mohafaza)] = demand.corr(civ_rank, method="pearson")

    # sort dictionary by value
    pearson_corr = dict(sorted(pearson_corr.items(), key=lambda kv: kv[1], reverse=True))
    subsets = list(pearson_corr.keys())
    values = list(pearson_corr.values())

    df = pd.DataFrame({'subsets': subsets, 'correlation': values})
    # df.plot.bar(x='subsets', y='correlation')
    sns.barplot(x="subsets", y="correlation", data=df)
    plt.xticks(fontsize=8, rotation=45)
    plt.title('Correlation between Demand and Civilians Rank for each Data Subset')
    fig = plt.gcf()
    fig.set_size_inches(12, 10)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + 'barplot.png')
    plt.close()


def demand_civ_dist_collated_boxplot(df, cols, output_folder, fig_name):
    df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
    fig, axarr = plt.subplots(len(cols), 1, figsize=(12, 8))
    sns.boxplot(y=cols[0], x="month_year", data=df, ax=axarr[0])
    axarr[0].tick_params(axis='x', rotation=45)
    sns.boxplot(y=cols[1], x="month_year", data=df, ax=axarr[1])
    axarr[1].tick_params(axis='x', rotation=45)
    sns.boxplot(y=cols[2], x="month_year", data=df, ax=axarr[2])
    axarr[2].tick_params(axis='x', rotation=45)
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(os.path.join(output_folder, '%s.png' % fig_name))
    plt.close()


if __name__ == '__main__':
    df_train_collated = pd.read_csv('../input/all_with_date/collated/all_columns/df_train_collated.csv')
    df_test_collated = pd.read_csv('../input/all_with_date/collated/all_columns/df_test_collated.csv')

    # box plots super imposing demand, civilians rank, and distance
    demand_civ_dist_collated_boxplot(df_train_collated, ['demand', 'civilians_rank', 'distance'], 'eda_plots/demand_civ_dist_collated/', 'df_train_collated')
    demand_civ_dist_collated_boxplot(df_test_collated, ['demand', 'civilians_rank', 'distance'], 'eda_plots/demand_civ_dist_collated/', 'df_test_collated')

    # box plot of demand in collated data
    boxplot_collated(df_train_collated, 'demand', 'eda_plots/demand_collated/', 'df_train_collated_demand')
    boxplot_collated(df_test_collated, 'demand', 'eda_plots/demand_collated/', 'df_test_collated_demand')

    # box plot of civilians rank in collated data
    boxplot_collated(df_train_collated, 'civilians_rank', 'eda_plots/civ_rank_collated/', 'civ_rank_collated', by_year=True)

    # box plot of demand in collated data
    boxplot_collated(df_train_collated, 'demand', 'eda_plots/demand_collated/', 'demand_all_collated', all=True)

    # eda for separated datasets
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']

    for service in services:
        for mohafaza in mohafazas:
            df = pd.read_csv('output/multivariate_datasubsets/%s_%s.csv' % (service, mohafaza))
            create_lineplot(df, ['demand', 'civilians_rank', 'distance'], 'eda_plots/demand_civ_dist/', '%s_%s' % (service, mohafaza))
            create_lineplot(df, ['demand', 'AverageTemp', 'AverageWindSpeed', 'Precipitation'], 'eda_plots/temp_wind_prec/', '%s_%s' % (service, mohafaza))
            corr_matrix(df, service, mohafaza, 'eda_plots/corr_matrix/')
            demand_boxplot(df, service, mohafaza, 'eda_plots/demand_boxplot/')
            corr_civ_demand('eda_plots/vdc_plots/')

