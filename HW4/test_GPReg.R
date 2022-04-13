library(Rcpp)
rm(list=ls())
sourceCpp('HW4/GaussianProcessReg.cpp')

betas <- c(rep(5, 200), rep(0, 200))

testcase = simul_dat_Probit(n=100, intercept=0.1, beta=betas, X_rho = 0.01, X_sd = 2)

result = Bayes_shrinkage_GP_reg(y=testcase$y, X=testcase$X)


