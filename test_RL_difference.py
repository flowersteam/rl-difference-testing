"""
Codebase for the paper: How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.
Here we implement three functions:

1) welch_test: performs Welch's t-test at significance level alpha. Wraps around ttest_ind function of scipy. See
function documentation.

2) bootstrap_test: performs bootstrap confidence interval test at significance level alpha. Wraps around bootstrapped
library from https://github.com/facebookincubator/bootstrapped. See function documentation.
3) empirical_false_pos_rate: computes the empirical false positive rate base on a sample of performance measures. See
function documentation.

4) compute_beta: computes the false negative rate: the probability to miss an underlying different of size epsilon,
given the Welch's t-test, at significance level alpha and the standard deviations of two algorithms. See function
documentation for details.

5) plot_beta: takes as input the output of compute_beta and allows to plot beta as a function of the sample size, for various
values of the effect size epsilon. See function documentation for details.

author: Cédric Colas
contact: cedric.colas@inria.fr
"""

import numpy as np
import scipy.stats as stats
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)


# load mock data
path_to_data1 = './data1'
path_to_data2 = './data2'
data1 = np.loadtxt(path_to_data1)
data2 = np.loadtxt(path_to_data2)

# Significance level to be used by both tests
alpha = 0.05
# Requirement in type-II error
beta_requirement = 0.2

# define the range of sample size to compute and plot beta
sample_size = range(2, 50)

# define the effect size epsilon. Here we define epsilon proportionally to smaller average performance
if data1.mean() < data2.mean():
    m_smaller = data1.mean()
else:
    m_smaller = data2.mean()
epsilon = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * m_smaller
epsilon = epsilon.tolist()


def welch_test(data1, data2, alpha=0.05, tail=2):
    """
    Wraps around ttest_ind function of scipy, without assuming equal variances.

    Params
    ------

    - data1 (ndarray of dim 1)
    The performance measures of Algo1.
    - data2 (ndarray of dim 1)
    The performance measures of Algo2.
    - alpha (float in ]0,1[)
    The significance level used by the Welch's t-test.
    - tail (1 or 2)
    Perform a one tail or two tail test.

    """
    assert tail==1 or tail==2, "tail should be one or two, referring to the one-sided or two-sided t-test."
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    assert alpha <1 and alpha >0, "alpha should be between 0 and 1"

    t, p = stats.ttest_ind(data1, data2, equal_var=False)

    if tail==1:
        alpha = 2*alpha
    if p <= alpha:
        if t<0:
            print("\n\nResult of the Welch's t-test at level %02g: μ2>μ1, the test passed with p-value = %02g." %(alpha, p))
        else:
            print("\n\nResult of the Welch's t-test level %02g: μ1>μ2, the test passed with p-value = %02g." %(alpha, p))
    else:
        print("\n\nResults of the Welch's t-test level %02g: there is not enough evidence to prove any order relation between μ1 and μ2." % alpha)
    print("Welch's t-test done.")


def bootstrap_test(data1, data2, alpha=0.05):
    """
    Wraps around bootstrap test from https://github.com/facebookincubator/bootstrapped/.

    Params
    ------

    - data1 (ndarray of dim 1)
    The performance measures of Algo1.
    - data2 (ndarray of dim 1)
    The performance measures of Algo2.
    - alpha (float in ]0,1[)
    The significance level used by the Welch's t-test.

    """
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    assert alpha <1 and alpha >0, "alpha should be between 0 and 1"

    res = bs.bootstrap_ab(data1, data2, bs_stats.mean, bs_compare.difference, alpha=alpha, num_iterations=10000)
    decision = np.sign(res.upper_bound) == np.sign(res.lower_bound)


    if decision:
        if np.sign(res.upper_bound)<0:
            print("\n\nResult of the bootstrap test at level %02g: μ2>μ1, the test passed with a confidence interval μ1-μ2 in %02g, %02g."
                  % (alpha, res.lower_bound, res.upper_bound))
        else:
            print("\n\nResult of the bootstrap test level %02g: μ1>μ2, the test passed with a confidence interval μ1-μ2 in %02g, %02g."
                  % (alpha, res.lower_bound, res.upper_bound))
    else:
        print("\n\nResults of the bootstrap test level %02g: there is not enough evidence to prove any order relation between μ1 and μ2." % alpha)
    print("Bootstrap test done.")



def empirical_false_pos_rate(data, alpha=0.05):
    """
    Compute and plot empirical estimates of the probability of type-I error given a list of performance measures.
    If this list is of size N_data
    This is done for N=2:floor(N_data/2). Two different tests are used: the bootstrap confidence interval test and the
    Welch's t-test, both with significance level alpha.

    Params
    ------
    - data1 (ndarray of dim 1)
    The performance measures of the considered algorithm.
    - alpha (float in ]0,1[)
    The significance level used by the two tests.
    """
    print('\n\nComputing empirical false positive rate ..')
    data = data.squeeze()
    sizes = range(2, data.size//2)
    nb_reps = 1000
    results = np.zeros([nb_reps, len(sizes), 2])
    blue = [0,0.447,0.7410,1]
    orange = [0.85,0.325,0.098,1]

    for i_n, n in enumerate(sizes):
        print('     N =', n)
        ind = list(range(2*n))
        for rep in range(nb_reps):
            # take two groups of size n in data, at random
            np.random.shuffle(ind)
            sample_1 = data[ind[:n]]
            sample_2 = data[ind[n:2*n]]
            # perform the two-tail Welch's t-test
            results[rep, i_n, 0] = stats.ttest_ind(sample_1, sample_2, equal_var=False)[1] < alpha
            # perform the bootstrap confidence interval test
            res_final = bs.bootstrap_ab(sample_1, sample_2, bs_stats.mean, bs_compare.difference, num_iterations=10000)
            results[rep, i_n, 1] = np.sign(res_final.upper_bound) == np.sign(res_final.lower_bound)

    res_mean = results.mean(axis=0)
    plt.figure(figsize=(16,10), frameon=False)
    plt.plot(sizes, alpha * np.ones(len(sizes)), c='k', linewidth=5, linestyle='--')
    plt.plot(sizes, res_mean[:,0], color=blue, linewidth=4)
    plt.plot(sizes, res_mean[:,1], color=orange, linewidth=4)

    plt.legend([u'α=%02d'%alpha] + ["Welch's $t$-test", 'bootstrap test'])
    plt.xlabel('sample size (N)')
    plt.ylabel('P(false positive)')
    plt.title(u'Estimation of type-I error rate as a function of $N$ when $α=0.05$')
    print("\n   Given N=%i and α=%02g, you can expect false positive rates: \n     For the Welch's t-test: %02g \n     For the bootstrap test: %02g."
          % (data.size //2, alpha, res_mean[-1,0], res_mean[-1,1] ))
    print('Done.')





def compute_beta(epsilon, sample_size, alpha=0.05, data1=None, data2=None, s1=None, s2=None, beta_requirement=0.2):
    """
    Computes the probability of type-II error (or false positive rate) beta to detect and effect size epsilon
    when testing for a difference between performances of Algo1 versus Algo2, using a Welch's t-test
    with significance alpha and sample size N.

    Params
    ------
    - epsilon (int, float or list of int or float)
    The effect size one wants to be able to detect.
    - sample_size (int or list of int)
    The sample size (assumed equal for both algorithms).
    - alpha (float in ]0,1[)
    The significance level used by the Welch's t-test.
    - data1 (ndarray of dim 1)
    The performance measures of Algo1. Optional if s1 is provided.
    - data2 (ndarray of dim 1)
    The performance measures of Algo2. Optional if s2 is provided.
    - s1 (float)
    The standard deviation of Algo1, optional if data1 is provided.
    - s2 (float)
    The standard deviation of Algo2, optional if data2 is provided.
    - beta_requirement (float in ]0,1[, optional)
    Requirements on the value of beta.
    """
    print('\n\nComputing the false negative rate as a function of sample size, for various effect sizes ..')
    assert alpha < 1 and alpha > 0, "alpha must be in ]0,1["
    assert data1 is not None or s1 is not None, "data1 or s1 should be provided"
    assert data2 is not None or s2 is not None, "data1 or s2 should be provided"

    if type(epsilon) is int or type(epsilon) is float:
        epsilon = [epsilon]
        n_eps = 1
    else:
        n_eps = len(epsilon)

    if type(sample_size) is int:
        sample_size = [sample_size]
        n_sample_size = 1
    else:
        n_sample_size = len(sample_size)

    if data1 is not None:
        s1 = data1.std(ddof=1)
    else:
        s1 = s1

    if data2 is not None:
        s2 = data2.std(ddof=1)
    else:
        s2 = s2



    results = np.zeros([n_sample_size, n_eps])
    t_dist = stats.distributions.t

    selected_sample_size = []
    for i_diff, eps in enumerate(epsilon):
        sample_size_found = False # True if a previous sample size satisfied beta requirements for the current epsilon
        for i_n, n in enumerate(sample_size):
            nu = (s1 ** 2 + s2 ** 2) ** 2 * (n - 1) / (s1 ** 4 + s2 ** 4)
            t_eps = eps / np.sqrt((s1 ** 2 + s2 ** 2) / n)
            t_crit = t_dist.ppf(1 - alpha, nu)
            results[i_n, i_diff] = t_dist.cdf(t_crit - t_eps, nu)
            if results[i_n, i_diff] < beta_requirement and not sample_size_found:
                sample_size_found = True
                selected_sample_size.append(str(n))
        if not sample_size_found:
            selected_sample_size.append('>'+str(n))

    eps_str = str()
    for i in range(n_eps):
        eps_str += '    ε = %0g  -->  N: %s \n ' % (epsilon[i], selected_sample_size[i])

    print('\nSample sizes satisfying β=%02g are:\n %s' % (beta_requirement, eps_str))
    print('Done.')


    return results



def plot_beta(beta, epsilon, sample_size, beta_requirement=0.2):
    """
    Plot the probability of type-II error beta as a function of the sample size, for various effect sizes epsilon

    Params
    ------
    - beta (ndarray of shape (size(N), size(epsilon))
    Contains values of beta for various epsilon and N
    - epsilon (int, float or list of int or float)
    The effect size one wants to be able to detect.
    - sample_size (int or list of int)
    The sample size (assumed equal for both algorithms).
    - beta_requirement (float in ]0,1[, optional)
    Requirements on the value of beta.
    """
    if type(epsilon) is int or type(epsilon) is float:
        epsilon = [epsilon]
        n_eps = 1
    else:
        n_eps = len(epsilon)

    if type(sample_size) is int:
        sample_size = [sample_size]
        n_sample_size = 1
    else:
        n_sample_size = len(sample_size)

    try:
        assert n_sample_size > 1
    except:
        print("Beta cannot be plotted as a function of only one sample size.")
        return

    legend = [u'$β_{requirement}$']
    plt.figure(figsize=(16,10), frameon=False)
    plt.plot(sample_size, beta_requirement * np.ones(n_sample_size), 'k', linewidth=3, linestyle='--')
    plt.plot(sample_size, beta, linewidth=2)
    if n_eps>1:
        legend += [u'ε = %02d' % epsilon[i] for i in range(n_eps)]
    else:
        legend += u'ε = %02d' % epsilon[0]
    plt.legend(legend)
    plt.xlabel('sample size (N)')
    plt.ylabel('P(false negative)')
    plt.title(u'Estimation of type-II error rate as a function of ε and $N$')


welch_test(data1, data2, alpha, tail=2)
bootstrap_test(data1, data2, alpha)
empirical_false_pos_rate(data1, alpha)
beta = compute_beta(epsilon, sample_size, alpha, data1, data2, beta_requirement=beta_requirement)
plot_beta(beta, epsilon, sample_size, beta_requirement=beta_requirement)
plt.show()
