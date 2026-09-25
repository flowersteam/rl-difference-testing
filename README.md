# rl-difference-testing
Simple tools for statistical analyses in RL experiments

Code base for the paper: How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.

The script can be run on mock data. It can also be imported without running the example analysis. There are 5 implemented functions:

1) `welch_test`: performs Welch's t-test at significance level alpha. It supports one-sided or two-sided alternative hypotheses through the `alternative` argument. Wraps around `ttest_ind` from scipy. See function documentation.

2) `bootstrap_test`: performs bootstrap confidence interval test at significance level alpha. Wraps around the bootstrapped library from https://github.com/facebookincubator/bootstrapped. See function documentation.

3) `empirical_false_pos_rate`: computes the empirical false positive rate based on a sample of performance measures. For Welch's t-test, the alternative hypothesis can be one-sided or two-sided. Statistical tests supposedly ensure a false positive rate of alpha, the significance level. However, computing empirical estimations of the false positive rate based on a set of empirical measures can lead to different conclusions. See the article for further details. This function automatically plots the empirical estimation of the false positive rate as a function of the sample size N, N varying from 2 to half the number of measures available.

4) `compute_beta`: computes the false negative rate, that is, the probability of missing an underlying difference epsilon between the performances of two algorithms, using a Welch's t-test with significance level alpha and given the standard deviations s1 and s2 of the two algorithms. The calculation can be performed for either a one-sided or two-sided alternative hypothesis. It also prints the theoretically required sample size to meet requirements on beta for a given effect size epsilon. This estimation might not be accurate because of inaccuracies in s1 and s2; see Section 5 of the article for further discussion. See function documentation for details.

5) `plot_beta`: takes as input the output of `compute_beta` and allows beta to be plotted as a function of sample size for various values of the effect size epsilon. See function documentation for details.

Author: Cédric Colas

Contact: cedric.colas@inria.fr
