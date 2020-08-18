import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
runit = robjects.r
runit['source']('smogn.R')


def get_rare(y, method='extremes', extr_type='both', thresh=0.8, coef=1.5, control_pts=None):
    #  yrel=None, phi_params=None, loss_params=None, df=None, target_variable=None,
    ''' we will be getting the relevance function on all the data not just the training data because
    when we want to apply Lime on the 'rare' testing instances, the relevance function must map all possible demand
    values to a certain relevance. If it happens that some demand values are present only in the testing
    and not in the training data, we cannot detect rare values correctly. The way we compute
    rare values depends on the relevance

    :param y: the target variable vector
    :param method: 'extremes' or 'range'. Default is 'extremes'
    :param extr_type: 'both', 'high', or 'low'
    :param thresh: threshold. Default is 0.8
    :param coef: parameter needed for method "extremes" to specify how far the wiskers extend to the most extreme data point in the boxplot. The default is 1.5.
    :param control_pts: if method == 'range', then this is the relevance matrix provided by the user. Default is None

    :return the indices of the rare values in the data
    '''

    yrel = get_relevance(y, df=None, target_variable=None, method=method, extr_type=extr_type, control_pts=control_pts)

    # get the the phi.control returned parameters that are used as input for computing the relevance function phi
    # (function provided by R UBL's package: https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi)
    # (function provided by R UBL's package
    # https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi.control)
    # we need those returned parameters for computing rare values

    print('relevance method - phi function : {}'.format(method))

    if control_pts is None:
        # without relevance matrix
        print('control.pts - phi function: {}'.format(control_pts))
        print('without relevance matrix')
        params = runit.get_relevance_params_extremes(y, rel_method=method, extr_type=extr_type, coef=coef)
    else:
        # with relevance matrix (provided by the user)
        print('control.pts - phi function: {}'.format(control_pts))
        print('with relevance matrix')
        params = runit.get_relevance_params_range(y, rel_method=method, extr_type=extr_type, coef=coef,
                                                  relevance_pts=control_pts)

    # phi params
    phi_params = params[0]
    loss_params = params[1]

    phi_params = dict(zip(phi_params.names, list(phi_params)))
    loss_params = dict(zip(loss_params.names, list(loss_params)))

    print('\nCONTROL PTS')
    print(phi_params['control.pts'])
    rare_indices = get_rare_indices(y=y, y_rel=yrel, thresh=thresh, controlpts=phi_params['control.pts'])
    # print('rare indices are: {}'.format(rare_indices))

    return rare_indices, phi_params, loss_params, yrel


def get_relevance(y, df=None, target_variable=None, method='extremes', extr_type='both', control_pts=None):
    '''
    gets the relevance values of the target variable vector
    :param y: the target variable vector
    :param df: if y in None, this must be passed. It is the data frame of interest
    :param target_variable: if y is None, this must be passed. It is the name of the target variable
    :param method: 'extremes' or 'range'
    :param extr_type: 'both', 'high', or 'low'
    :param control_pts: if method == 'range', will be a relevance matrix provided by the user
    :return: the relevance values of the associated target variable
    '''

    # get the target variable vector y
    if y is None:
        if df is None or target_variable is None:
            raise ValueError('if y is None, neither df nor target_variable must be None')
        y = df[target_variable]

    # check that the passed parameters are in order
    if method != 'range' and method != 'extremes':
        raise ValueError('method must be "range" or "extremes", there is no method called "%s"' % method)
    elif method == 'range' and control_pts is None:
        raise ValueError('If method == "range", then control_pts must not be None')
    elif method == 'extremes' and extr_type not in ['high', 'low', 'both']:
        raise ValueError('extr_type must wither be "high", "low", or "both"')
    else:
        if control_pts is None:
            print('getting yrel - Control pts is {}, method is {}'.format(control_pts, method))
            y_rel = runit.get_yrel(y=np.array(y), meth=method, extr_type=extr_type)
        else:
            print('getting yrel - Control pts is not None, method is {}'.format(method))
            y_rel = runit.get_yrel(y=np.array(y), meth=method, extr_type=extr_type, control_pts=control_pts)

    return y_rel


def get_rare_indices(y, y_rel, thresh, controlpts):
    '''
    get the indices of the rare values in the data
    :param y: the target variable vector
    :param y_rel: the target variable (y) relevance vector
    :param thresh: the threshold of interest
    :param controlpts: the phi.control (function provided by R UBL's package: https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi.control)
    returned parameters that are used as input for computing the relevance function phi (function provided by R UBL's package: https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi)
    :return: the indices of the rare values in 'y'
    '''

    # references
    # https://github.com/paobranco/SMOGN-LIDTA17/blob/8964a2327de19f6ca9e6f7055479ca863cd6b8a0/R_Code/ExpsDIBS.R#L41

    # transform controlpts returned by R into a python list
    controlpts = list(np.array(controlpts))
    # print(controlpts)

    # boolean variable indicating whether both low and high rare exist
    both = [controlpts[i] for i in [1, 7]] == [1, 1]

    # initialize rare cases to empty list (in case there are no rare cases at all)
    rare_cases = []

    if both:
        # bothr = True
        print('\nWe have both low and high extremes')
        rare_low = [i for i, e in enumerate(y_rel) if e > thresh and y[i] < controlpts[3]]
        rare_high = [i for i, e in enumerate(y_rel) if e > thresh and y[i] > controlpts[3]]

        # merge two lists (of low rare + high rare) together
        rare_cases = rare_low + rare_high

    else:
        print('\nWe dont have both', end=' ')
        if controlpts[1] == 1:
            print('We have only low rare')
            # lowr = True
            rare_cases = [i for i, e in enumerate(y_rel) if e > thresh and y[i] < controlpts[3]]
        else:
            print('We have only high rare')
            # highr = True
            rare_cases = [i for i, e in enumerate(y_rel) if e > thresh and y[i] > controlpts[3]]

    total = len(rare_cases)

    print('Total Number of rare cases: %d out of %d' % (total, len(y)))
    print('Percentage of Rare Cases: %.2f%%\n' % (total/len(y) * 100))

    return rare_cases


def plot_relevance(y, yrel, target_variable, output_folder, fig_name):
    '''
    plots the relevance of the target variable
    :param y: vector of the target variable
    :param yrel: vector of the relevance values of y
    :param target_variable: name of the target variable column
    :param output_folder: path to the output folder. If its not there it will be created dynamically at run time
    :param figname: name of the figure. Plot will be saved as such.
    :return:
    '''

    reldict = {}
    for i, e in enumerate(y):
        if e not in reldict:
            reldict[e] = yrel[i]

    reldict = dict(collections.OrderedDict(sorted(reldict.items())))
    plt.plot(list(reldict.keys()), list(reldict.values()))
    plt.xlabel(target_variable)
    plt.ylabel('relevance')

    check_create_output_folder(output_folder)
    fig_name = check_figname(fig_name)

    plt.savefig(output_folder + fig_name)
    plt.close()


def plot_rare(y, rare_cases, target_var_name, output_folder, fig_name, model_name=None):
    '''
    plots the rare values in the target variable
    :param y: target variable vectors
    :param rare_cases: list of indices of rare cases
    :param target_var_name: target variable column name
    :param output_folder: path to the output folder
    :param fig_name: name of the figure (to be saved as such)
    :return: plot of the target variable with rare values marked in red
    '''

    plt.plot(range(len(y)), y, c='blue', label='%s' % target_var_name)
    plt.scatter(rare_cases, y[rare_cases], label='rare values', c='red', marker='d')
    plt.legend()
    plt.title('Line plot of {} and rare values'.format(target_var_name))

    check_create_output_folder(output_folder)

    if model_name is None:
        fig_name = check_figname(fig_name)
    else:
        fig_name = check_figname(fig_name + '_' + model_name)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.legend()

    plt.savefig(output_folder + fig_name)
    plt.close()

    print(output_folder + fig_name)


def plot_density(df, target_variable, output_folder, fig_name, title, model_name=None):
    ''' produces a density plot of a given variable '''
    sns.distplot(df[target_variable], hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label=target_variable).set(title=title, xlabel=target_variable, ylabel='density')

    # create the output folder if it does not exist already
    check_create_output_folder(output_folder)

    if model_name is None:
        fig_name = check_figname(fig_name)
    else:
        fig_name = check_figname(fig_name + '_' + model_name)
    print(output_folder + fig_name)
    plt.savefig(output_folder + fig_name)
    plt.close()


def plot_target_variable(df, target_variable, output_folder, fig_name):
    y = df[target_variable]
    plt.plot(list(range(len(y))), sorted(y))

    check_create_output_folder(output_folder)
    fig_name = check_figname(fig_name)

    plt.xlabel('Index')
    plt.ylabel(target_variable)

    plt.savefig(output_folder + fig_name)
    plt.close()


def check_figname(figname):
    '''
    adds the extension '.png' to the name of the image
    :param figname: name of the image. To be saved as such
    :return: the name with '.png' added if it is not there
    '''
    if figname[-4:] != '.png':
        figname = figname + '.png'
    return figname


def check_create_output_folder(output_folder):
    ''' checks if the output folder specified exists, creates it if not '''
    if not os.path.exists(output_folder):
        if output_folder[-1] != '\\':
            output_folder = output_folder + '\\'
        os.makedirs(output_folder)

