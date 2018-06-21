# rl-difference-testing
Simple tools for statistical analyses in RL experiments

Code base for the paper: How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.

The script can be run on mock data. There are 5 implemented functions:

1) welch_test: performs Welch's t-test at significance level alpha. Wraps around ttest_ind function of scipy. See
function documentation.

2) bootstrap_test: performs bootstrap confidence interval test at significance level alpha. Wraps around bootstrapped
library from https://github.com/facebookincubator/bootstrapped. See function documentation.

3) empirical_false_pos_rate: computes the empirical false positive rate based on a sample of performance measures. See
function documentation. Statistical tests supposedly ensure the false positive rate to alpha, the significance level. However, computing empirical estimations of the false positive rate based on a set of empirical measures can lead to different conclusions. See the article for further details. This function automatically plots the empirical estimation of the false positive rate as a function of the sample size N, N varying from 2 to half the number of measures available.

4) compute_beta: computes the false negative rate, that is to say, the probability to miss an underlying difference epsilon between the performances of two algorithms (s1,s2), using a Welch's t-test with significance level alpha and given the standard deviations of both algorithms. It also print the theoretically required sample size to meet requirements in beta, considering an effect size epsilon. This estimation might not be accurate because of inaccuracies on s1 and s2, see Section 5 of the article for further discussion. See function
documentation for details.

5) plot_beta: takes as input the output of compute_beta and allows to plot beta as a function of the sample size, for various
values of the effect size epsilon. See function documentation for details.

Author: Cédric Colas

Contact: cedric.colas@inria.fr
