# rl-difference-testing
Simple tools for statistical analyses in RL experiments

Code base for the paper: How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.

The script can be run on mock data. There are 5 functions implemented:

1) welch_test: performs Welch's t-test at significance level alpha. Wraps around ttest_ind function of scipy. See
function documentation.

2) bootstrap_test: performs bootstrap confidence interval test at significance level alpha. Wraps around bootstrapped
library from https://github.com/facebookincubator/bootstrapped. See function documentation.

3) empirical_false_pos_rate: computes the empirical false positive rate base on a sample of performance measures. See
function documentation. Statistical tests supposedly ensure the false positive rate to alpha, the significance level. However, computing empirical estimations of the false positive rate based on a set of empirical measures can lead to different conclusions. See the article for further details. This function automatically plot the empirical estimations of the false positive rate as a function of the sample size N, N varying between 2 and half the number of measures that are available.

4) compute_beta: computes the false negative rate: the probability to miss an underlying different of size epsilon,
given the Welch's t-test, at significance level alpha and the standard deviations of two algorithms. It also print the theoretically required sample size to meet requirements in beta considering effect size epsilon. Please note this estimation might not be accurate, see Section 5 of the article for further discussion. See function
documentation for details.

5) plot_beta: takes as input the output of compute_beta and allows to plot beta as a function of the sample size, for various
values of the effect size epsilon. See function documentation for details.

author: Cédric Colas
contact: cedric.colas@inria.fr
