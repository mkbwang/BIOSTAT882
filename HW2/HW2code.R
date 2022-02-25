library(Rcpp)
library(RcppArmadillo)
library(coda)

rm(list=ls())
sourceCpp('HW2/BayesProbitReg.cpp')

mnist_train_x = read.csv('HW2/mnist_100_train_X.csv') |> as.matrix()
mnist_train_y = read.csv('HW2/mnist_100_train_Y.csv') |> as.matrix()

initial_beta = rep(c(0,0),length=ncol(mnist_train_x))
initial_sigma2_beta = 1000
include_intercept = TRUE
initial_intercept = 0.0

mcmc_sample = 2000

burnin_HMC = 20000
burnin_MALA = 40000
burnin_RW = 400000
burnin_Gibbs = 1000

thinning_HMC = 10
thinning_MALA = 20
thinning_RW = 100
thinning_Gibbs = 3

step_adjust_accept = 100
maxiter_adjust_accept_HMC = 40000
maxiter_adjust_accept_MALA = 80000
maxiter_adjust_accept_RW = 600000


RW_start <- proc.time()
RW_fit <- Bayes_Probit_reg(mnist_train_y, mnist_train_x, method="RW",
                           initial_beta, 
                           initial_sigma2_beta, include_intercept,
                           initial_intercept, mcmc_sample, burnin_RW, thinning_RW, step_adjust_accept,maxiter_adjust_accept_RW,
                           target_accept = 0.25, initial_step_size = 1e-8, a_gamma = 0.01, b_gamma=0.01, verbose=10000)
RW_duration <- proc.time() - RW_start

plot(RW_fit$trace$loglik, type='l')
plot(RW_fit$trace$accept_rate, type='l')
RW_effsizes <- effectiveSize(t(RW_fit$mcmc$beta))

save(RW_fit, RW_duration, file='HW2/RW_fit_2000.RData')

MALA_start <- proc.time()
MALA_fit <- Bayes_Probit_reg(mnist_train_y, mnist_train_x, method="MALA",
                             initial_beta, 
                             initial_sigma2_beta, include_intercept,
                             initial_intercept, mcmc_sample, burnin_MALA, thinning_MALA, step_adjust_accept,maxiter_adjust_accept_MALA,
                             target_accept = 0.5, initial_step_size = 1e-6, a_gamma = 0.01, b_gamma=0.01)
MALA_duration <- proc.time() - MALA_start


plot(MALA_fit$trace$loglik, type='l')
plot(MALA_fit$trace$accept_rate, type='l')
MALA_effsizes <- effectiveSize(t(MALA_fit$mcmc$beta))
mean(MALA_effsizes)
save(MALA_fit, MALA_duration, file='HW2/MALA_fit_2000.RData')


HMC_start <- proc.time()
HMC_fit <- Bayes_Probit_reg(mnist_train_y, mnist_train_x, method="HMC",
                            initial_beta, 
                            initial_sigma2_beta, include_intercept,
                            initial_intercept, mcmc_sample, burnin_HMC, thinning_HMC, step_adjust_accept, maxiter_adjust_accept_HMC,
                            target_accept = 0.7, initial_step_size = 1e-4, a_gamma = 0.01, b_gamma=0.01)
HMC_duration <- proc.time() - HMC_start


plot(HMC_fit$trace$loglik, type='l')
plot(HMC_fit$trace$accept_rate, type='l')
HMC_effsizes <- effectiveSize(t(HMC_fit$mcmc$beta))
mean(HMC_effsizes)
save(HMC_fit, HMC_duration, file='HW2/HMC_fit_2000.RData')

Gibbs_start <- proc.time()
Gibbs_fit <- Bayes_Probit_reg(mnist_train_y, mnist_train_x, method="Gibbs",
                              initial_beta, 
                              initial_sigma2_beta, include_intercept,
                              initial_intercept, mcmc_sample, burnin_Gibbs, thinning_Gibbs, step_adjust_accept, maxiter_adjust_accept_HMC,
                              target_accept = 0.7, initial_step_size = 1e-5, a_gamma = 0.01, b_gamma=0.01, verbose=100)
Gibbs_duration <- proc.time() - Gibbs_start
plot(Gibbs_fit$trace$loglik, type='l')
plot(Gibbs_fit$trace$accept_rate, type='l')
Gibbs_effsizes <- effectiveSize(t(Gibbs_fit$mcmc$beta))
mean(Gibbs_effsizes)

save(Gibbs_fit, Gibbs_duration, file='HW2/Gibbs_model_2000.RData')
