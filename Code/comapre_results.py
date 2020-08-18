import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set(style="whitegrid")


def show_separated_best(dir, output_folder, dir_name):
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']
    rmses = {}
    for service in services:
        for mohafaza in mohafazas:
            errors = pd.read_csv(dir + '%s_%s/' % (service, mohafaza) + 'error_metrics_csv/errors.csv')
            errors = errors.sort_values(by='rmse')
            errors.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
            rmses['%s_%s' % (service, mohafaza)] = errors.iloc[0, errors.columns.get_loc('rmse')]

    df = pd.DataFrame({'datasubset': list(rmses.keys()), 'rmse': list(rmses.values())})
    df = df.sort_values(by='rmse')
    # ax = df.plot.barh(x='datasubset', y='rmse')
    ax = sns.barplot(x='rmse', y='datasubset', data=df)
    # for i, v in enumerate(list(df['rmse'])):
    #     ax.text(v + 0.5, i + .25, str('%.1f' % v), color='blue', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    # ax.invert_yaxis()
    plt.yticks(rotation=30)
    plt.title(dir_name)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + '%s_best.png' % dir_name)
    plt.close()


def show_collated_results(dir, output_folder):
    errors1 = pd.read_csv('../output/grid/%s/train_collated_test_collated/error_metrics_csv/errors.csv' % dir)
    errors1 = errors1.drop(['coefficients'], axis=1)
    errors2 = pd.read_csv('../output/neural_networks/%s/train_collated_test_collated/error_metrics_csv/errors.csv' % dir)
    errors = pd.concat([errors1, errors2])

    errors = errors.sort_values(by='rmse')
    errors.rename(columns={'Unnamed: 0': 'model'}, inplace=True)
    # ax = errors.plot.barh(x='model', y='rmse')
    ax = sns.barplot(x='rmse', y='model', data=errors)
    # ax.barh(errors['model'], errors['rmse'])
    # for j, v in enumerate(list(errors['rmse'])):
    #     ax.text(v + 0.5, j + .25, str('%.1f' % v), color='blue', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title('%s' % dir)
    # ax.invert_yaxis()
    plt.yticks(rotation=30)
    plt.title(dir)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + '%s.png' % dir)
    plt.close()


if __name__ == '__main__':

    # show the model's performance for each collated directory
    # ['all_columns', 'all_columns_minus_weather', 'all_columns_minus_weather_minus_lags']
    dirs = ['all_columns', 'all_columns_minus_weather', 'all_columns_minus_weather_minus_lags', 'all_columns_minus_weather_minus_vdc']
    for dir in dirs:
        output_folder = '../output/collated_plots/'
        show_collated_results(dir, output_folder)

    # show the best RMSE's attained for each datasubset in each separated directory
    # ['all_columns', 'all_columns_minus_weather', 'all_columns_minus_weather_minus_lags']
    for dir in dirs:
        path = '../output/grid/%s/train_separated_test_separated/' % dir
        output_folder = '../output/separated_plots/'
        name = dir
        show_separated_best(path, output_folder, name)

