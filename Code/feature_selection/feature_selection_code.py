import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import boxcox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

# variables that control the number of features to be selected in each FS method
NUM_FEATURES_UNIVARIATE = 10  # This should be less than the total number of input features
NUM_FEATURES_RFE = 10  # This should be less than the total number of input features
NUM_FEATURES_PCA = 2  # This should be less than the total number of input features
import warnings

warnings.filterwarnings("ignore")

"""
    User edits in this file
    --------------------

    lines 16, 17, and 18 in order to control the number of features to be selected in each feature selection method


    Scaling
    --------------------

    * scale=True, scale_input=True, scale_output=True: Will scale both the input and the output columns.
    * scale=True, scale_input=True, scale_output=False: Will scale only the input columns.
    * scale=True, scale_input=False, scale_output=True: Will scale only the output column.
    * scale=True, scale_input=False, scale_output=False: Will not scale any columns, although scale=True, but either scale_input or scale_output must be True
    * scale=False, scale_input=True, scale_output=True: Will not scale any columns, although both scale_input=True and scale_output=True, but scale must be True as well in order to perform any scaling exercise.
    * scale=False, scale_input=True, scale_output=False: Will not scale any columns, although scale_input=True, but scale must be True as well in order to perform any scaling exercise.
    * scale=False, scale_input=False, scale_output=True: Will not scale any columns, although scale_output=True, but scale must be True as well in order to perform any scaling exercise.
    * scale=False, scale_input=False, scale_output=False: Will not scale any columns. 


    Indexing Input Columns in order to Scale
    --------------------------------------------

    If scale=True and scale_input=True
        * input_zscore=(start_index_1, end_index_1): will apply Z-score scaling for the input columns starting at index start_index_1 and ending at end_index_1 (eclusive). By defualt, None. If None, no z-score scaling to any of the input columns is applied.
        * input_minmax=(start_index_2, end_index_2): will apply min-max scaling for the input columns starting at index start_index_2 and ending at end_index_2 (eclusive). By defualt, None. If None, no min-max scaling to any of the input columns is applied.
        * input_box=(start_index_3, end_index_3): will apply box-cox transformation for the input columns starting at index start_index_3 and ending at end_index_3 (eclusive). By defualt, None. If None, no box-cox transformation to any of the input columns is applied.
        * input_log=(start_index_4, end_index_4): will apply log transformation for the input columns starting at index start_index_4 and ending at end_index_4 (eclusive). By defualt, None. If None, no log transformation to any of the input columns is applied.


    Specifying Scaling Type to the Output Column
    --------------------------------------------

    If scale=True and scale_output=True:
        * output_zscore: Boolean, by default, False. If True, Z-score scaling for the output column will be applied.
        * output_minmax: Boolean, by default, False. If True, min-max scaling for the output column will be applied.
        * output_box: Boolean, by default, False. If True, box-cox transformation for the output column will be applied.
        * output_log: Boolean, by default, False. If True, log transformation for the output column will be applied.

    Note: Either one of the above must be True, and all others must be False because we have to apply only one kind of scaling for the output column.

    Saving Feature Selection Plots
    --------------------

    * output_folder: the path to the output folder that will be holding several modeling plots. If the path specified does not exist, it will be created dynamically at runtime.

    Columns
    --------------------

    * cols_drop: list containing the names of the columns the user wants to drop from the data. By default, None. If None, no columns will be dropped from the data.
    * target_variable: name of the column holding the target variable (this will be the output column)


    Raises
    --------------------
    ValueError
        * If NUM_FEATURES_UNIVARIATE (line 16 in feature_selection.py)is greater than the total number of input features.
    ValueError
        * If NUM_FEATURES_RFE (line 17 in feature_selection.py) is greater than the total number of input features.
    ValueError
        * If NUM_FEATURES_PCA (line 19 in feature_selection.py) is greater than the total number of input features.


    Feature Selection Methods:
    --------------------------

    * drop_zero_std(): drops columns that have 0 standard deviation. (Actually will not drop but show the 
    columns that must be dropped)

    * drop_low_var(): drops columns that have low variance. (Actually will not drop but show the 
    columns that must be dropped)

    * drop_high_correlation(): drops columns that have high correlation. (Actually will not drop but show the 
    columns that must be dropped)

    * feature_importance(xg_boost=True, extra_trees=False): Applies feature importance to the data.
        * xg_boost=True, extra_trees=False: will perform feature importance using XG Boost only
        * xg_boost=False, extra_trees=True: will perform feature importance using Extra Trees only
        * xg_boost=True, extra_trees=True: will perform feature importance using both XG Boost and Extra Trees
        * xg_boost=False, extra_trees=False: Nothing will happen. Avoid this if you want to use feature selection.
        * Default Behavior: i.e. if we do: feature_importance() it will do only XG Boost. As: xg_boost=False, extra_trees=True

    * univariate(): Applies Univariate Feature Selection with NUM_FEATURES_UNIVARIATE being selected (specified in line 17 in feature_selection.py). Raises Vlaue Error in this is greater than the total number of input features.

    * rfe():  Applies Recursive Feature Elimination with NUM_FEATURES_RFE being selected (specified in line 18 in feature_selection.py). Raises Vlaue Error in this is greater than the total number of input features.

    * pca():  Applies PCA NUM_FEATURES_PCA principal components done. (specified in line 19 in feature_selection.py). Raises Vlaue Error in this is greater than the total number of input features.

"""


class FeatureSelection:

    def __init__(self, df, target_variable, output_folder, cols_drop=None,
                 scale=True, scale_input=True, scale_output=False,
                 output_zscore=False, output_minmax=True, output_box=False, output_log=False,
                 input_zscore=None, input_minmax=None, input_box=None, input_log=None):

        # drop un-wanted columns from the data
        if cols_drop is not None:
            df = df.drop(cols_drop, axis=1)

        # drop NaN values
        df = df.dropna()

        # features/columns names (without including target variable)
        self.feature_names = list(df.drop(target_variable, axis=1).columns.values)

        # define input and output
        X = np.array(df.loc[:, df.columns != target_variable])
        y = np.array(df.loc[:, target_variable])

        X_df = df.loc[:, df.columns != target_variable]
        y_df = df.loc[:, target_variable]

        self.df = df
        self.target_variable = target_variable
        self.output_folder = output_folder

        self.X = X
        self.y = y

        self.X_df = X_df
        self.y_df = y_df

        # scaling input & output
        self.scale = scale
        self.scale_input = scale_input
        self.scale_output = scale_output

        # specify scaling method for output
        self.output_zscore = output_zscore
        self.output_minmax = output_minmax
        self.output_box = output_box
        self.output_log = output_log

        # specify scaling method for input
        self.input_zscore = input_zscore
        self.input_minmax = input_minmax
        self.input_box = input_box
        self.input_log = input_log

        # lists that will store the index of the columns to scale
        self.idx_zscore, self.idx_minmax, self.idx_box, self.idx_log = None, None, None, None

        self.labelsdict = {
            'demand': 'demand',
            'civilians_rank': 'civilians',
            'distance': 'dist',
            'AverageTemp': 'Avg.Temp',
            'AverageWindSpeed': 'Avg.WS',
            'Precipitation': 'precip',
            'w_{t-1}': 't-1',
            'w_{t-2}': 't-2',
            'w_{t-3}': 't-3',
            'w_{t-4}': 't-4',
            'w_{t-5}': 't-5',
            'w_{t-1}_trend': 'trend',
            'w_{t-1}_seasonality': 'season',
            'service_General Medicine': 'Gen.Med',
            'service_Gynaecology': 'Gynaecol',
            'service_Pediatrics': 'Ped',
            'service_Pharmacy': 'Pharm',
            'mohafaza_B': 'mB',
            'mohafaza_N': 'mN',
            'mohafaza_NE': 'mNE',
        }

        if scale:
            # if we want to scale
            if input_zscore is not None:
                self.idx_zscore = list(range(input_zscore[0], input_zscore[1]))
            if input_minmax is not None:
                self.idx_minmax = list(range(input_minmax[0], input_minmax[1]))
            if input_box is not None:
                self.idx_box = list(range(input_box[0], input_box[1]))
            if input_log is not None:
                self.idx_log = list(range(input_log[0], input_log[1]))

            self.X, self.y = self.scale_cols()

    def drop_zero_std(self):
        """
        function that removes features having 0 standard deviation
        Note: This function performs analysis using all the features (with target varoable included)
        """
        print('\n********** Method 1: Calculate the no of features which has standard deviation as zero. **********\n')
        # Remove Constant Features
        df = self.df
        constant_features = [feat for feat in df.columns if df[feat].std() == 0]
        if not constant_features:
            print('We did not find any features having std of 0')
            print("data shape remains: {}".format(df.shape))
            return df
        else:
            print('The following columns have 0 std: {}. They will be removed'.format(constant_features))
            df.drop(labels=constant_features, axis=1, inplace=True)
            print("Original data shape: {}".format(df.shape))
            print("Reduced data shape: {}".format(df.shape))
            return df

    def drop_low_var(self, variance_threshold=0.18):
        """
        function that drops the columns having low variance
        Note: This function performs analysis using all the features (with target variable included)
        """
        print('\n********** Method 2: Calculate the no of features which has low variance. **********\n')
        df = self.df
        features = list(df.columns.values)
        sel = VarianceThreshold(threshold=variance_threshold)
        sel.fit(df)
        mask = sel.get_support()
        reduced_df = df.loc[:, mask]
        selected_features = []
        for i in range(len(features)):
            if mask[i]:
                selected_features.append(features[i])
        dropped_features = set(features) - set(selected_features)
        print("Original data shape- ", df.shape)
        print("Reduced feature dataset shape-", reduced_df.shape)
        print("Dimensionality reduced from {} to {}.".format(df.shape[1], reduced_df.shape[1]))
        print('Selected features: {}. Len: {}'.format(selected_features, len(selected_features)))
        print('Dropped features: {}. Len: {}'.format(dropped_features, len(dropped_features)))

        return reduced_df

    def drop_high_correlation(self, moph_project=False):
        '''
        drops columns that have high correlation with each other
        :param moph_project: boolean, indicating if this is for this particular project. If yes,
        it will replace columns names by shorter abbreviations
        :return:
        '''
        df = self.df
        output_folder = self.output_folder

        # check if output folder exists, if not create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # labels to replace columns names by for shorter representation
        if moph_project:
            df = df.rename(columns=self.labelsdict)

        # the correlation matrix plot
        corr = df.corr()
        # Add the mask to the heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool), 1)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_ylim(20.0, 0)
        heatmap = sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f", ax=ax)
        heatmap = heatmap.get_figure()
        heatmap.set_size_inches(18.5, 10.5)

        # rotate x ticks
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)

        plt.title('Correlation matrix')
        plt.savefig(output_folder + 'corr_matrix', dpi=100)
        plt.close()

        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(6)

        # the columns with high correlation part

        # redefine the data frame (with the original namings - not the abbreviations)
        df = self.df
        corr_matrix = df.corr().abs()
        # Create a True/False mask and apply it
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        tri_df = corr_matrix.mask(mask)
        # List column names of highly correlated features (r >0.5 )
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.5)]
        # Drop the features in the to_drop list
        reduced_df = df.drop(to_drop, axis=1)
        print("The reduced_df dataframe has {} columns".format(reduced_df.shape[1]))
        print("Dropped columns: {}".format(to_drop))


    # def drop_high_correlation(self):
    #     """
    #     function that drops columns having high correlation
    #     Note: This function performs analysis using all the features (with target variable included)
    #     """
    #     print('\n********** Method 3: Remove the features which have a high correlation. **********\n')
    #     df = self.df
    #     corr = df.corr()
    #     # mask = np.triu(np.ones_like(corr, dtype=bool))
    #
    #     # Add the mask to the heatmap
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #
    #     # ax.set_ylim(20.0, 0)
    #     # heatmap = sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f", ax=ax)
    #     heatmap = sns.heatmap(corr, center=0, linewidths=1, annot=True, fmt=".2f", ax=ax)
    #     heatmap = heatmap.get_figure()
    #
    #     output_folder = self.output_folder
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     heatmap.set_size_inches(18.5, 10.5)
    #     heatmap.savefig(output_folder + 'corr_matrix.png')
    #
    #     corr_matrix = df.corr().abs()
    #
    #     # Create a True/False mask and apply it
    #     mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    #     tri_df = corr_matrix.mask(mask)
    #
    #     # List column names of highly correlated features (r >0.5 )
    #     to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.5)]
    #
    #     # Drop the features in the to_drop list
    #     reduced_df = df.drop(to_drop, axis=1)
    #     print("The reduced_df dataframe has {} columns".format(reduced_df.shape[1]))
    #     print("Dropped Columns: {}".format(to_drop))
    #
    #     plt.close()

    def feature_importance(self, xg_boost=True, extra_trees=False):
        """
        function that displays feature importance using XG-Boost and Extra Trees
        Note: This function performs analysis using X and y
        * xg_boost=True, extra_trees=False: will perform feature importance using XG Boost only
        * xg_boost=False, extra_trees=True: will perform feature importance using Extra Trees only
        * xg_boost=True, extra_trees=True: will perform feature importance using both XG Boost and Extra Trees
        * xg_boost=False, extra_trees=False: Nothing will happen. Avoid this if you want to use feature selection.
        """
        output_folder = self.output_folder
        feature_names = self.feature_names

        X = self.X_df
        y = self.y_df

        if xg_boost:
            print('\n********** Method 4: Calculating the feature importance using XGBoost. **********\n')
            ''' feature importance using XGBoost '''
            feature_names = feature_names
            housing_dmatrix = xgb.DMatrix(X, y, feature_names=feature_names)
            # Create the parameter dictionary: params
            params = {"objective": "reg:squarederror", "max_depth": "4"}
            # Train the model: xg_reg
            xg_reg = xgb.train(dtrain=housing_dmatrix, params=params, num_boost_round=10)

            feature_imp = dict(
                sorted(xg_reg.get_score(importance_type='weight').items(), key=lambda kv: kv[1], reverse=True))
            print('\nFeatures - Importance\n')
            for key, value in feature_imp.items():
                print('%s: %.5f' % (key, value))
            print('\n')

            # Plot the feature importances
            xgb.plot_importance(xg_reg)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            fig = plt.gcf()
            fig.set_size_inches(15, 10.5)
            plt.title('XGBoost Feature Importance')
            fig.savefig(output_folder + 'xgb_fs', dpi=100)
            plt.close()
            print('saved plot in {}/{}'.format(output_folder, 'xgb_fs'))

        if extra_trees:
            print('\n********** Method 5: Calculating the feature importance using Extra Trees. **********\n')
            model = ExtraTreesRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            feature_imp = {}
            for i in range(len(model.feature_importances_)):
                # print('%s: %.5f' % (columns[i], model.feature_importances_[i]))
                feature_imp[feature_names[i]] = model.feature_importances_[i]
            feature_imp = dict(sorted(feature_imp.items(), key=lambda kv: kv[1], reverse=True))
            print('\nFeatures - Importance\n')
            for key, value in feature_imp.items():
                print('%s: %.5f' % (key, value))
            print('\n')
            # print(model.feature_importances_)
            # use inbuilt class feature_importances of tree based classifiers
            # plot graph of feature importances for better visualization
            feat_importances = pd.Series(model.feature_importances_, index=X.columns)
            feat_importances.nlargest(20).plot(kind='barh')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            fig = plt.gcf()
            fig.set_size_inches(15, 10.5)
            plt.title('Extra Trees Feature Importance')
            fig.savefig(output_folder + 'extratrees_fs.png', dpi=100)
            plt.close()
            print('saved plot in {}/{}'.format(output_folder, 'extratrees_fs.png'))

    def univariate(self, moph_project=False):
        ''' univariate feature selection '''

        if NUM_FEATURES_UNIVARIATE > self.X.shape[1]:
            raise ValueError('NUM_FEATURES_UNIVARIATE must be less than the total number of input columns.'
                             '\n. Please change line 18 in feature_selection.py')
        print('\n********** Method 6: Univariate Feature Selection **********\n')
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        X = self.X_df
        y = self.y_df
        labels = self.labelsdict
        # labels to replace columns names by for shorter representation
        if moph_project:
            del labels[self.target_variable]
            X = X.rename(columns=self.labelsdict)

        output_folder = self.output_folder

        univariate = f_regression(X, y)

        # Capture P values in a series
        univariate = pd.Series(univariate[1])
        univariate.index = X.columns
        univariate.sort_values(ascending=False, inplace=True)
        # Plot the P values
        univariate.sort_values(ascending=False).plot.bar(figsize=(20, 8))
        # plt.show()
        plt.yscale("log")
        plt.xticks(rotation=45)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.title('Univariate Feature Selection')
        plt.savefig(output_folder + 'univariate_fs')
        plt.close()

        k_best_features = SelectKBest(f_regression, k=NUM_FEATURES_UNIVARIATE).fit(X.fillna(0), y)
        print(list(X.columns[k_best_features.get_support()]))

        X_train = k_best_features.transform(X.fillna(0))
        print(X_train.shape)

    def rfe(self):
        '''
        Recursive feature elimination
        :param k: top k features returned
        :return:
        '''

        if NUM_FEATURES_RFE > self.X.shape[1]:
            raise ValueError('NUM_FEATURES_RFE must be less than the total number of input columns.'
                             '\n. Please change line 19 in feature_selection.py')

        print('\n********** Method 7: RFE **********\n')

        X = self.X
        y = self.y

        estimator = Lasso()
        selector = RFE(estimator, NUM_FEATURES_RFE)
        fit = selector.fit(X, y)
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)
        selected = []
        for i in range(len(fit.support_)):
            if fit.support_[i]:
                selected.append(self.feature_names[i])
        print('Selected Features: ', selected)

    # def pca(self):
    #     '''
    #     principal components analysis
    #     :param k: top k components
    #     :return:
    #     '''
    #
    #     if NUM_FEATURES_PCA > self.X.shape[1]:
    #         raise ValueError('NUM_FEATURES_PCA must be less than the total number of input columns.'
    #                          '\n. Please change line 20 in feature_selection.py')
    #
    #     print('\n********** Method 8: PCA **********\n')
    #
    #     X = self.X
    #
    #     print(X.shape)
    #
    #     df = self.df
    #     print(len(df))
    #
    #     # get quartiles of the data
    #     desc = df[self.target_variable].describe()
    #
    #     # quartiles
    #     min = desc['min']
    #     quart = desc['25%']
    #     half = desc['50%']
    #     three = desc['75%']
    #     max = desc['max']
    #
    #     df['group'] = pd.cut(df[self.target_variable], bins=[min, quart, half, three, max],
    #                          labels=['min-25%', '25%-50%', '50%-75%', '75%-max'])
    #
    #     pca = PCA(n_components=NUM_FEATURES_PCA)
    #     principal_components = pca.fit_transform(X)
    #
    #     principalDf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    #
    #     df.reset_index(drop=True, inplace=True)
    #     principalDf.reset_index(drop=True, inplace=True)
    #
    #     finalDf = pd.concat([principalDf, df[['group']]], axis=1)
    #
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.set_xlabel('PC1', fontsize=15)
    #     ax.set_ylabel('PC2', fontsize=15)
    #     ax.set_title('2 component PCA', fontsize=20)
    #     targets = ['min-25%', '25%-50%', '50%-75%', '75%-max']
    #     colors = ['r', 'g', 'b', 'k']
    #     for target, color in zip(targets, colors):
    #         indicesToKeep = finalDf['group'] == target
    #         ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
    #                    , finalDf.loc[indicesToKeep, 'PC2']
    #                    , c=color
    #                    , s=50)
    #     ax.legend(targets)
    #     ax.grid()
    #
    #     output_folder = self.output_folder
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     plt.savefig(output_folder + 'pca_plot')
    #     plt.close()
    #
    #     self.pca_biplot(principal_components[:, 0:2], np.transpose(pca.components_[0:2, :]), df['group'],
    #                     self.feature_names)
    #     fig = plt.gcf()
    #     fig.set_size_inches(20, 10.5)
    #     fig.savefig(output_folder + 'pca_biplot')
    #     plt.close()
    #
    # def pca_biplot(self, score, coeff, y, labels=None):
    #     xs = score[:, 0]
    #     ys = score[:, 1]
    #     n = coeff.shape[0]
    #     scalex = 1.0 / (xs.max() - xs.min())
    #     scaley = 1.0 / (ys.max() - ys.min())
    #     plt.scatter(xs * scalex, ys * scaley, c='c')
    #     for i in range(n):
    #         plt.arrow(0, 0, coeff[i, 0] * 2, coeff[i, 1] * 2,
    #                   color='k', alpha=0.5)
    #         if labels is None:
    #             plt.text(coeff[i, 0] * 2.3, coeff[i, 1] * 2.3, "Var" + str(i + 1), color='k', fontweight='bold', ha='center',
    #                      va='center')
    #         else:
    #             plt.text(coeff[i, 0] * 2.3, coeff[i, 1] * 2.3, labels[i], color='k', fontweight='bold', ha='center', va='center')
    #     plt.xlim(-1, 1)
    #     plt.ylim(-1, 1)
    #     plt.xlabel("PC{}".format(1))
    #     plt.ylabel("PC{}".format(2))
    #     plt.grid()

    def scale_cols(self):
        X = self.X
        y = self.y
        if self.input_zscore is not None:
            # apply Standard scaling to the specified columns.
            scaler = StandardScaler()
            X = X.astype('float64')

            X_zscaled = scaler.fit_transform(X[:, self.idx_zscore])

            for i in range(len(self.idx_zscore)):
                X[:, self.idx_zscore[i]] = X_zscaled[:, i]

            if self.scale_output:
                if self.output_zscore:
                    scaler_out = StandardScaler()
                    y = y.reshape(-1, 1)
                    y = scaler_out.fit_transform(y)
                    y = y.reshape(-1)

        if self.input_minmax is not None:
            # apply MinMax scaling to the specified columns.
            scaler = MinMaxScaler()

            if X.dtype != 'float64':
                X = X.astype('float64')

            X_minmaxscaled = scaler.fit_transform(X[:, self.idx_minmax])

            for i in range(len(self.idx_minmax)):
                X[:, self.idx_minmax[i]] = X_minmaxscaled[:, i]

            if self.scale_output:
                if self.output_minmax:
                    scaler_out = MinMaxScaler()

                    y = y.reshape(-1, 1)
                    y = scaler_out.fit_transform(y)
                    y = y.reshape(-1)

        if self.input_box is not None:
            # apply BoxCox transform to the specified columns.

            if X.dtype != 'float64':
                X = X.astype('float64')

            X_boxscaled = np.array([list(boxcox(X[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T

            for i in range(len(self.idx_box)):
                X[:, self.idx_box[i]] = X_boxscaled[:, i]

            if self.scale_output:
                if self.output_box:
                    y, _ = boxcox(y)

        if self.input_log is not None:
            # apply Log transform to the specified columns.

            if X.dtype != 'float64':
                X = X.astype('float64')

            X_logscaled = np.log(X[:, self.idx_log])

            for i in range(len(self.idx_log)):
                X[:, self.idx_log[i]] = X_logscaled[:, i]

            if self.scale_output:
                if self.output_log:
                    y = np.log(y)

        return X, y

