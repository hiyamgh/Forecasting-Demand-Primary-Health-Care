import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
runit = robjects.r
runit['source']('smogn.R')


def evaluate(actual, predicted, thresh, target_variable, df=None, df_train=None, df_test=None, method='extremes', extr_type='high', coef=1.5, control_pts=None):
    '''
    produces regression and utility based error metrics
    :param actual: vector of the actual values
    :param predicted: vector of the predicted values
    :param thresh: threshold of relevance
    :param target_variable: name of the target variable
    :param df: data frame representing the whole dataset. By default None
    :param df_train: data frame representing the training dataset. By default None. If None, df must be provided
    :param df_test: data frame representing the testing dataset. By default None. If None, df must be provided
    :param method: 'extremes' or 'range'
    :param extr_type: 'high', 'low', or 'both'
    :param coef:
    :param control_pts: if method is 'range', it is the relevance matrix. By default None
    :return: prints regression and utility based error metrics of teh provided actual and predicted
    '''
    if df is None:
        if df_train is None or df_test is None:
            raise ValueError('df is None. You must provide both df_train and df_test')
        else:
            df = pd.concat([df_train, df_test])

    if control_pts is None:
        # without relevance matrix
        params = runit.get_relevance_params_extremes(df[target_variable], rel_method=method, extr_type=extr_type, coef=coef)
    else:
        # with relevance matrix (provided by the user)
        params = runit.get_relevance_params_range(df[target_variable], rel_method=method, extr_type=extr_type, coef=coef,
                                                  relevance_pts=control_pts)

    # phi and loss params
    phi_params = params[0]
    loss_params = params[1]

    phi_params = dict(zip(phi_params.names, list(phi_params)))
    loss_params = dict(zip(loss_params.names, list(loss_params)))

    nb_columns = len(list(df_test.columns.values)) - 1
    errors = get_stats(actual, predicted, nb_columns, thresh, phi_params, loss_params)

    return errors


def get_stats(y_test, y_pred, nb_columns, thr_rel, phi_params, loss_params):
    '''
    Function to compute regression and utility based error metrics between actual and predicted values as well
    as their correlation
    :param y_test: vector of the actual values
    :param y_pred: vector of the predicted values
    :param nb_columns: number of columns <<discarding the target variable column>>
    :param thr_rel: threshold of relevance
    :param phi_params: phi function parameters
    :param loss_params: loss parameters
    :return: R2, Adj-R2, RMSE, MSE, MAE, MAPE, F2, F1, F0.5, precision, recall, pearson, spearman, distance
    '''

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    if not isinstance(y_test, list):
        y_test = list(y_test)
    if not isinstance(y_pred, list):
        y_pred = list(y_pred)

    n = len(y_test)

    r2_Score = r2_score(y_test, y_pred) # r-squared
    adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1) # adjusted r-squared
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE
    mse_score = mean_squared_error(y_test, y_pred) # MSE
    mae_score = mean_absolute_error(y_test, y_pred) # MAE
    mape_score = mean_absolute_percentage_error(y_test, y_pred) # MAPE

    trues = np.array(y_test)
    preds = np.array(y_pred)

    method = phi_params['method']
    npts = phi_params['npts']
    controlpts = phi_params['control.pts']
    ymin = loss_params['ymin']
    ymax = loss_params['ymax']
    tloss = loss_params['tloss']
    epsilon = loss_params['epsilon']

    rmetrics = runit.eval_stats(trues, preds, thr_rel, method, npts, controlpts, ymin, ymax, tloss, epsilon)

    # create a dictionary of the r metrics extracted above
    rmetrics_dict = dict(zip(rmetrics.names, list(rmetrics)))

    if isinstance(y_pred[0], np.ndarray):
        y_pred_new = [x[0] for x in y_pred]
        y_pred = y_pred_new
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    distance_corr = distance.correlation(y_test, y_pred)

    print('\nUtility Based Metrics')
    print('F1: %.5f' % rmetrics_dict['ubaF1'][0])
    print('F2: %.5f' % rmetrics_dict['ubaF2'][0])
    print('F05: %.5f' % rmetrics_dict['ubaF05'][0])
    print('precision: %.5f' % rmetrics_dict['ubaprec'][0])
    print('recall: %.5f' % rmetrics_dict['ubarec'][0])

    print('\nRegression Error Metrics')
    print('R2: %.5f' % r2_Score)
    print('Adj-R2: %.5f' % adjusted_r2)
    print('RMSE: %.5f' % rmse_score)
    print('MSE: %.5f' % mse_score)
    print('MAE: %.5f' % mae_score)
    print('MAPE: %.5f' % mape_score)

    print('\nCorrelations')
    print('Pearson: %.5f' % pearson_corr)
    print('Spearman: %.5f' % spearman_corr)
    print('Distance: %.5f' % distance_corr)

    results = []
    # for e in [['ubaF1'][0], rmetrics_dict['ubaF2'][0], rmetrics_dict['ubaF05'][0], rmetrics_dict['ubaprec'][0], rmetrics_dict['ubarec'][0]]:
    #     results.append(e)
    results.append(r2_Score)
    results.append(adjusted_r2)
    results.append(rmse_score)
    results.append(mse_score)
    results.append(mae_score)
    results.append(mape_score)
    results.append(pearson_corr)
    results.append(spearman_corr)
    results.append(distance_corr)

    return results, rmetrics_dict


if __name__ == '__main__':
    # sample usage example

    df_train = pd.read_csv('../input/all_without_date/collated/all_columns/df_train_collated.csv')
    df_test = pd.read_csv('../input/all_without_date/collated/all_columns/df_test_collated.csv')

    prediction = pd.read_csv('../output/khalil/all_columns.csv')
    prediction = prediction.drop('Unnamed: 0', axis=1)

    actual = prediction['demand']
    predicted = prediction['predicted']

    # if we want to make a relevance matrix, we do as follows:
    # * The first column indicates the y values of interest
    # * The second column indicates a mapped value of relevance, either 0 or 1, where 0 is the least relevant
    # and 1 is the most relevant,
    # * The third column is indicative. It will be adjusted afterwards, use 0 in most cases.
    rel_mat2 = np.array([[100, 0, 0], [380, 1, 0], [400, 1, 0]])

    results, ubrmetrics = evaluate(actual=actual, predicted=predicted, thresh=0.8, target_variable='demand',
             df=None, df_train=df_train, df_test=df_test,
             method='range', extr_type='high', coef=1.5, control_pts=rel_mat2)

