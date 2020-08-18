import pandas as pd
from Code.feature_selection_code import FeatureSelection
import os

# read the training + testing data
df_train_collated = pd.read_csv('../input/all_without_date/collated/all_columns/df_train_collated.csv')
df_test_collated = pd.read_csv('../input/all_without_date/collated/all_columns/df_test_collated.csv')

# concatenate
df = pd.concat([df_train_collated, df_test_collated])

# abs_path = os.getcwd().replace('\\', '/').split('Code/')[0]
# output_folder = abs_path + 'output_ubr_shallow_corrected/feature_selection/'
output_folder = '../output/feature_selection/non_ubr/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# don't change the name of the target variable (as it will be taken automatically from the loop)
fs = FeatureSelection(df, target_variable='demand',
                      output_folder=output_folder,
                      cols_drop=None,
                      scale=True,
                      scale_input=True,
                      scale_output=False,
                      output_zscore=False,
                      output_minmax=False,
                      output_box=False,
                      output_log=False,
                      input_zscore=None,
                      input_minmax=(0, 12),
                      input_box=None,
                      input_log=None)

fs.drop_zero_std()
fs.drop_low_var()
fs.drop_high_correlation(moph_project=True)
fs.feature_importance(xg_boost=True, extra_trees=True)
fs.univariate(moph_project=True)
fs.rfe()
