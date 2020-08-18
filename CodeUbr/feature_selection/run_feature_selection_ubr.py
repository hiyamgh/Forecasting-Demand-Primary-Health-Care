import pandas as pd
from Code.feature_selection_code import FeatureSelection
import os

# read the training + testing data
# smogn_0.95_Euclidean

# train: normalized, test: not normalized
# we know that the lowest value of demand is 1
# get the highest value of demand from test
# do the minmax myself on the test data, provided the min and max
# I will not use sklearn's minmax because it has to be fitted
# and if I fit it to the testing data only it will interpret min
# and max wrong because it will take min and max only from
# the testing data whilst it must take them from both combined
# IDEA:
# 1. take the min and max from original combined
# 2. do that for each column
# 3. normalize that column in the testing


def get_minmax(df, col):
    return min(list(df[col])), max(list(df[col]))


def normalize_data(df, df_orig):
    # df = df.reset_index(drop=True)
    for col in list(df_orig.columns.values):
        # print('{}: {}'.format(col, df.columns.get_loc(col)))
        res = []
        if col != 'demand':
            minm, maxm = get_minmax(df_orig, col)
            for i, row in df.iterrows():
                val = row[col]
                # df.loc[i, df.columns.get_loc(col)] =
                res.append((val - minm) / (maxm - minm))
            df[col] = res
    return df


def rename_cols(df):
    df = df.rename(columns={
            'w__t_1_': 'w_{t-1}',
            'w__t_2_': 'w_{t-2}',
            'w__t_3_': 'w_{t-3}',
            'w__t_4_': 'w_{t-4}',
            'w__t_5_': 'w_{t-5}',
            'w__t_1__trend': 'w_{t-1}_trend',
            'w__t_1__seasonality': 'w_{t-1}_seasonality',
            'service_General_Medicine': 'service_General Medicine'
        })
    return df


# path to input train & test datasets
input_df_train = 'E:/moph_final_final_outputs/all_columns_balance/all_columns_balance/smogn_0.95_Euclidean/grid_search/oversampled_dataset/df_train_linear_svr_smogn.csv'
input_df_test = 'E:/moph_final_final_outputs/all_columns_balance/all_columns_balance/smogn_0.95_Euclidean/grid_search/shuffled_data/df_test_collated_shuffled.csv'
df_train_collated = rename_cols(pd.read_csv(input_df_train))
df_test_collated = rename_cols(pd.read_csv(input_df_test))

# get the original non-smogned data
dftr = pd.read_csv('../input/all_without_date/collated/all_columns/df_train_collated.csv')
dfte = pd.read_csv('../input/all_without_date/collated/all_columns/df_test_collated.csv')
df = pd.concat([dftr, dfte])

df_test_collated = normalize_data(df_test_collated, df)
df_combined = pd.concat([df_train_collated, df_test_collated])
df_combined = df_combined.reset_index(drop=True)

# # abs_path = os.getcwd().replace('\\', '/').split('Code/')[0]
# # output_folder = abs_path + 'output_ubr_shallow_corrected/feature_selection/'

output_folder = '../output/feature_selection/ubr/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# will not scale input cz data is already scaled
fs = FeatureSelection(df_combined, target_variable='demand',
                      output_folder=output_folder,
                      cols_drop=None,
                      scale=False,
                      scale_input=False,
                      scale_output=False,
                      output_zscore=False,
                      output_minmax=False,
                      output_box=False,
                      output_log=False,
                      input_zscore=None,
                      input_minmax=None,
                      input_box=None,
                      input_log=None)

fs.drop_zero_std()
fs.drop_low_var()
fs.drop_high_correlation(moph_project=True)
fs.feature_importance(xg_boost=True, extra_trees=True)
fs.univariate(moph_project=True)
fs.rfe()

df_combined.to_csv(os.path.join(output_folder, 'df_smogned.csv'), index=False)