rm(list = ls(all = TRUE))
Rcpp::sourceCpp("HW3/BayesLinearReg.cpp")

set.seed(2022)
betas <- c(-1,1,-1,0,0,0,0,0,0,0)
dat <- simul_dat_linear(n = 50, intercept = 1, beta = betas,
                        X_rho = 0.9, X_sd = 4, sigma_sq = 10)
system.time(glm_fit <- glm(dat$y~dat$X, family = "gaussian"))


system.time(gibbs_fit <- Bayes_linear_reg(dat$y,dat$X,method="Gibbs",
                                          initial_sigma_sq = 4,
                                          initial_sigma_beta_sq  = 4,
                                          include_intercept = TRUE,
                                          mcmc_sample = 1000,
                                          burnin = 5000,
                                          thinning = 10,
                                          max_iter = 1000,
                                          paras_diff_tol = 1e-4,
                                          ELBO_stop = 1,
                                          ELBO_diff_tol = 1e-4,
                                          verbose = 1000))

system.time(mfvb_fit <- Bayes_linear_reg(dat$y, dat$X, method = "MFVB",
                                         initial_sigma_sq = 1000,
                                         initial_sigma_beta_sq  = 1000,
                                         include_intercept = TRUE,
                                         mcmc_sample = 1000,
                                         burnin = 5000,
                                         thinning = 10,
                                         max_iter = 1000,
                                         paras_diff_tol = 1e-8,
                                         ELBO_stop = 0,
                                         ELBO_diff_tol = 1e-8,
                                         verbose = 1000,
                                         save_profile = 1))



Rmse <- c(mfvb =  mean((mfvb_fit$post_mean$beta - c(dat$intercept, dat$beta))^2),
          gibbs = mean((gibbs_fit$post_mean$beta - c(dat$intercept, dat$beta))^2),
          glm = mean((glm_fit$coefficients - c(dat$intercept, dat$beta))^2))

print(Rmse)
with(gibbs_fit$trace,plot(iters,logpost,type="l"))
