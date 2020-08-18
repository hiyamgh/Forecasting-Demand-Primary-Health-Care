import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import os
import pickle
import scipy

plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')

dest = '../output/statistical_distribution/'
if not os.path.exists(dest):
    os.makedirs(dest)


# Create models from data
def best_fit_distribution(data, bins=200, ax=None, sortby='p_value'):
    """Model data by finding best fit distribution to data"""

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # these are the only available distributions for ngboost
    # so we will see which one best fits our data
    # DISTRIBUTIONS = ['norm', 'lognorm', 'expon']
    DISTRIBUTIONS = ['norm', 'lognorm', 'expon']

    results = pd.DataFrame()
    sses = []
    pvals = []
    Ds = []
    parameters = []
    chi_square = []
    AIC = []
    BIC = []

    dist_param = {}
    size = len(data)

    # I don't see why in the link here: https://pythonhealthcare.org/2018/05/03/81-distribution-fitting-to-data/
    # they do the KS test on the standardized data
    # also, they use the standardized data for calculating chi squared ?

    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    percentile_bins = np.linspace(0, 100, 51)
    percentile_cutoffs = np.percentile(data, percentile_bins)
    observed_frequency, bins = (np.histogram(data, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Estimate distribution parameters from data
    for dist_name in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                distribution = getattr(scipy.stats, dist_name)

                # fit dist to data
                print('fitting {}'.format(dist_name))
                params = distribution.fit(data)
                parameters.append(params)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # from wikipedia
                # BIC = kln(n)-2ln(L)
                # AIC = 2k - 2ln(L)

                # calculate the log-likelihood
                LLH = distribution.logpdf(data, *params).sum()
                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * LLH
                bic = k * np.log(n) - 2 * LLH

                # Calculate fitted PDF and error with fit in distribution
                print('calculating stats')
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                D, pks = scipy.stats.kstest(y, dist_name, args=params)
                # dist_results.append((dist_name, D, pks, sse))
                sses.append(sse)
                pvals.append(pks)
                Ds.append(D)
                AIC.append(aic)
                BIC.append(bic)

                dist_param[dist_name] = params

                print('calculating chi squared')
                # Get expected counts in percentile bins
                # This is based on a 'cumulative distrubution function' (cdf)
                cdf_fitted = distribution.cdf(percentile_cutoffs, *params[:-2], loc=params[-2],
                                              scale=params[-1])
                expected_frequency = []
                for bin in range(len(percentile_bins) - 1):
                    expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
                    expected_frequency.append(expected_cdf_area)

                # calculate chi-squared
                expected_frequency = np.array(expected_frequency) * size
                cum_expected_frequency = np.cumsum(expected_frequency)
                ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
                chi_square.append(ss)

                print('\nDistribution: {}:'.format(dist_name))
                print('p-value: {}\nSSE: {}\nD: {}\nchi-squared: {}\nAIC: {}\nBIC: {}\n'.format(pks, sse, D, ss, aic, bic))
                print('--------------------------------------------------')

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax, label=dist_name)
                        plt.legend()
                        print('added plot for {}'.format(dist_name))
                except Exception:
                    pass

        except Exception:
            pass

    print('\nsaving results in a data frame ...')
    results['distribution'] = DISTRIBUTIONS
    results['chi_square'] = chi_square
    results['SSE'] = sses
    results['AIC'] = AIC
    results['BIC'] = BIC
    results['KS'] = Ds
    results['p_value'] = pvals
    results['params'] = parameters

    # sort results by p-value, sse, and chi-squared
    print('sorting results by: {}'.format(sortby))
    # descending: p-value, ascending: SSE, chi-square
    if sortby == 'p_value':
        results = results.sort_values(by=sortby, ascending=False).reset_index(drop=True)
    else:
        results = results.sort_values(by=sortby, ascending=True).reset_index(drop=True)

    best_distribution = results.iloc[0]['distribution']

    print('\nbest distribution: {}'.format(best_distribution))

    results.to_csv(os.path.join(dest, 'results_{}.csv'.format(sortby)), index=False)
    print('saved results_{}.csv in {}'.format(sortby, dest))

    return results, dist_param


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


# Load data from statsmodels datasets
# data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
df_train = pd.read_csv('../input/all_without_date/collated/all_columns/df_train_collated.csv')
df_test = pd.read_csv('../input/all_without_date/collated/all_columns/df_test_collated.csv')
df = pd.concat([df_train, df_test])
y = df['demand']
data = y

# Plot for comparison
plt.figure(figsize=(12, 8))
ax = data.plot(kind='hist', bins=50, density=True, color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], alpha=0.5)
# Save plot limits
dataYLim = ax.get_ylim()

sortings = ['SSE', 'AIC', 'BIC']

count = 0
bestresults = []

# get the best results from sorting by chi square (by default)
results, dist_param = best_fit_distribution(data, 200, ax, sortby='chi_square')
best_fit_name = results.iloc[0]['distribution']
best_fit_params = dist_param[best_fit_name]
print('\n\n\ndist_param: {}'.format(best_fit_params))
print('dist_param from data: {}'.format(results.iloc[0]['params']))
bestresults.append((best_fit_name, best_fit_params))

# get the best results by sorting by other metrics (sse, AIC, BIC)
for sorting in sortings:
    results = results.sort_values(by=sorting).reset_index(drop=True)
    results.to_csv(os.path.join(dest, 'results_{}.csv'.format(sorting)), index=False)
    print('saved results_{}.csv in {}'.format(sorting, dest))

    best_fit_name = results.iloc[0]['distribution']
    best_fit_params = dist_param[best_fit_name]
    bestresults.append((best_fit_name, best_fit_params))

    # Find best fit distribution
    best_dist = getattr(st, best_fit_name)

    if count == 0:
        # Update plots
        ax.set_ylim(dataYLim)
        ax.set_title(u'All Fitted Distributions')
        ax.set_xlabel(u'Demand')
        ax.set_ylabel('Frequency')
        plt.savefig(os.path.join(dest, 'all_dists.png'))
        plt.legend()
        plt.close()

    count += 1

    # Make PDF with best params
    pdf = make_pdf(best_dist, best_fit_params)

    # Display
    plt.figure(figsize=(12, 8))
    ax = pdf.plot(lw=2, label=best_fit_name, legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_title(u'best fit distribution \n' + dist_str)
    ax.set_xlabel(u'Demand')
    ax.set_ylabel('Frequency')
    plt.savefig(os.path.join(dest, 'best_fit_{}.png'.format(sorting)))
    plt.close()


# get the unique winning distributions to know the size of the qq_pp stacked plot
dists_unique = list(set([t[0] for t in bestresults]))
fig, axs = plt.subplots(len(dists_unique), 2)

i = 0
size = len(data)
data = sorted(data.to_list())
distsseen = []
for t in bestresults:
    # Set up distribution
    distribution = t[0]
    if distribution not in distsseen:
        distsseen.append(distribution)
        dist = getattr(scipy.stats, distribution)
        param = t[1]

        # Get random numbers from distribution
        norm = dist.rvs(*param[0:-2], loc=param[-2], scale=param[-1], size=size)
        norm.sort()

        # qq plot

        if axs.shape == (2,):
            axs[0].plot(list(norm), data, "o", color='b')
            min_value = np.floor(min(min(norm), min(data)))
            max_value = np.ceil(max(max(norm), max(data)))
            axs[0].plot([min_value, max_value], [min_value, max_value], 'r--')
            axs[0].set_xlim(min_value, max_value)
            axs[0].set_xlabel('Theoretical quantiles')
            axs[0].set_ylabel('Observed quantiles')
            title = 'qq plot for ' + distribution + ' distribution'
            axs[0].set_title(title)

            # pp plot

            # Calculate cumulative distributions
            bins = np.percentile(norm, range(0, 101))
            data_counts, bins = np.histogram(data, bins)
            norm_counts, bins = np.histogram(norm, bins)
            cum_data = np.cumsum(data_counts)
            cum_norm = np.cumsum(norm_counts)
            cum_data = cum_data / max(cum_data)
            cum_norm = cum_norm / max(cum_norm)

            # plot
            axs[1].plot(cum_norm, cum_data, "o", color='b')
            min_value = np.floor(min(min(cum_norm), min(cum_data)))
            max_value = np.ceil(max(max(cum_norm), max(cum_data)))
            axs[1].plot([min_value, max_value], [min_value, max_value], 'r--')
            axs[1].set_xlim(min_value, max_value)
            axs[1].set_xlabel('Theoretical cumulative distribution')
            axs[1].set_ylabel('Observed cumulative distribution')
            title = 'pp plot for ' + distribution + ' distribution'
            axs[1].set_title(title)

        else:
            axs[i, 0].plot(list(norm), data, "o", color='b')
            min_value = np.floor(min(min(norm), min(data)))
            max_value = np.ceil(max(max(norm), max(data)))
            axs[i, 0].plot([min_value, max_value], [min_value, max_value], 'r--')
            axs[i, 0].set_xlim(min_value, max_value)
            axs[i, 0].set_xlabel('Theoretical quantiles')
            axs[i, 0].set_ylabel('Observed quantiles')
            title = 'qq plot for ' + distribution + ' distribution'
            axs[i, 0].set_title(title)

            # pp plot

            # Calculate cumulative distributions
            bins = np.percentile(norm, range(0, 101))
            data_counts, bins = np.histogram(data, bins)
            norm_counts, bins = np.histogram(norm, bins)
            cum_data = np.cumsum(data_counts)
            cum_norm = np.cumsum(norm_counts)
            cum_data = cum_data / max(cum_data)
            cum_norm = cum_norm / max(cum_norm)

            # plot
            axs[i, 1].plot(cum_norm, cum_data, "o", color='b')
            min_value = np.floor(min(min(cum_norm), min(cum_data)))
            max_value = np.ceil(max(max(cum_norm), max(cum_data)))
            axs[i, 1].plot([min_value, max_value], [min_value, max_value], 'r--')
            axs[i, 1].set_xlim(min_value, max_value)
            axs[i, 1].set_xlabel('Theoretical cumulative distribution')
            axs[i, 1].set_ylabel('Observed cumulative distribution')
            title = 'pp plot for ' + distribution + ' distribution'
            axs[i, 1].set_title(title)

            i += 1
    else:
        continue
# Display plot
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.tight_layout(pad=4)
plt.savefig(os.path.join(dest, 'qq_pp.png'))
plt.close()
