rm(list = ls(all = TRUE))
Rcpp::sourceCpp("HW3/BayesLinearReg.cpp")

set.seed(2022)

## Scenario 1
betas1 <- (-1)^seq(1, 50)
dat1 <- simul_dat_linear(n = 1000, intercept = 1, beta = betas1,
                        X_rho = 0, X_sd = 1, sigma_sq = 1)
system.time(glm_fit1 <- glm(dat1$y~dat1$X, family = "gaussian"))


system.time(gibbs_fit1 <- Bayes_linear_reg(dat1$y,dat1$X,method="Gibbs",
                                          initial_sigma_sq = 1000,
                                          initial_sigma_beta_sq  = 1,
                                          include_intercept = TRUE,
                                          mcmc_sample = 500,
                                          burnin = 500,
                                          thinning = 10,
                                          verbose = 1000))

system.time(mfvb_fit1 <- Bayes_linear_reg(dat1$y, dat1$X, method = "MFVB",
                                         initial_sigma_sq = 1000,
                                         initial_sigma_beta_sq  = 1,
                                         include_intercept = TRUE,
                                         max_iter = 1000,
                                         paras_diff_tol = 1e-8,
                                         ELBO_stop = 0,
                                         ELBO_diff_tol = 1e-8,
                                         verbose = 1000,
                                         save_profile = 1))



Rmse1 <- c(mfvb =  mean((mfvb_fit1$post_mean$beta - c(dat1$intercept, dat1$beta))^2),
          gibbs = mean((gibbs_fit1$post_mean$beta - c(dat1$intercept, dat1$beta))^2),
          glm = mean((glm_fit1$coefficients - c(dat1$intercept, dat1$beta))^2))

print(Rmse1)
with(gibbs_fit1$trace,plot(iters,logpost,type="l"))
with(mfvb_fit1$trace,plot(iters,ELBO,type="l"))



## Scenario 2
set.seed(2022)
betas2 <-c(c(-1, 1, -1),rep(0, 7))
dat2 <- simul_dat_linear(n = 50, intercept = 1, beta = betas2,
                         X_rho = 0.9, X_sd = 1, sigma_sq = 10)

system.time(glm_fit2 <- glm(dat2$y~dat2$X, family = "gaussian"))

system.time(gibbs_fit2 <- Bayes_linear_reg(dat2$y,dat2$X,method="Gibbs",
                                           initial_sigma_sq = 1000,
                                           initial_sigma_beta_sq  = 1,
                                           include_intercept = TRUE,
                                           mcmc_sample = 500,
                                           burnin = 3000,
                                           thinning = 10,
                                           verbose = 1000))

system.time(mfvb_fit2 <- Bayes_linear_reg(dat2$y,dat2$X,method="MFVB",
                                           initial_sigma_sq = 1000,
                                           initial_sigma_beta_sq  = 1,
                                           include_intercept = TRUE,
                                           max_iter = 1000,
                                           paras_diff_tol = 1e-8,
                                           ELBO_stop = 1,
                                           ELBO_diff_tol = 1e-8,
                                           verbose = 1000))

Rmse2 <- c(mfvb =  mean((mfvb_fit2$post_mean$beta - c(dat2$intercept, dat2$beta))^2),
           gibbs = mean((gibbs_fit2$post_mean$beta - c(dat2$intercept, dat2$beta))^2),
           glm = mean((glm_fit2$coefficients - c(dat2$intercept, dat2$beta))^2))


with(gibbs_fit2$trace,plot(iters,logpost,type="l"))
with(mfvb_fit2$trace,plot(iters,ELBO,type="l"))
