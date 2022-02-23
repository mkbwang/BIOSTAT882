rm(list=ls(all=TRUE))
Rcpp::sourceCpp("HW2/BayesProbitReg.cpp")

p = 10
n = 500

true_beta = rep(c(-1,1),length=p)
true_intercept = 1


initial_beta = rep(c(0,0),length=p)
initial_sigma2_beta = 1000
include_intercept = TRUE
initial_intercept = 0.0
mcmc_sample = 500
burnin = 10000
thinning = 10
step_adjust_accept = 100
maxiter_adjust_accept = 5000
a_beta = 0.01
b_beta = 0.01


dat <- simul_dat_Probit(n, intercept = true_intercept, beta = true_beta, X_rho = 0.5, X_sd = 1)
glm_fit <-glm(dat$y~dat$X,family = binomial(link = "probit"))

HMC_fit <- Bayes_Probit_reg(dat$y,dat$X, method="HMC",
                           initial_beta, initial_sigma2_beta, include_intercept,
                           initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                           target_accept = 0.65, initial_step_size = 0.01, a_gamma = 0.01, b_gamma=0.01, leapfrog_steps = 20)

MALA_fit <- Bayes_Probit_reg(dat$y,dat$X, method="MALA",
                            initial_beta, initial_sigma2_beta, include_intercept,
                            initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                            target_accept = 0.50, initial_step_size = 0.01, a_gamma = 0.01, b_gamma=0.01)

RW_fit <- Bayes_Probit_reg(dat$y,dat$X, method="RW",
                            initial_beta, 
                            initial_sigma2_beta, include_intercept,
                            initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept,maxiter_adjust_accept,
                            target_accept = 0.25, initial_step_size = 0.01, a_gamma = 0.01, b_gamma=0.01)

gibbs_fit <- Bayes_Probit_reg(dat$y,dat$X, method="Gibbs",
                            initial_beta,
                            initial_sigma2_beta, include_intercept,
                            initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept,maxiter_adjust_accept,
                            target_accept = 0.25, initial_step_size = 0.01, a_gamma = 0.01, b_gamma=0.01)

plot(RW_fit$trace$accept_rate,type="l")
par(mfcol=c(1,1))
plot(RW_fit$trace$logpost,type="l",ylim=c(-1400,0))
lines(MALA_fit$trace$logpost,col="red")
lines(HMC_fit$trace$logpost,col="blue")

par(mfcol=c(1,3))
hist(MALA_fit$mcmc$sigma2_beta,xlim=c(0,10),ylim=c(0,300))
hist(HMC_fit$mcmc$sigma2_beta,xlim=c(0,10),ylim=c(0,300))
hist(RW_fit$mcmc$sigma2_beta,xlim=c(0,10),ylim=c(0,300))


#legend("bottomright",c("RW","MALA","HMC"),lty=1,col=c("black","red","blue"))
sum((MALA_fit$post_mean$beta - c(dat$intercept,dat$beta))^2)
sum((HMC_fit$post_mean$beta - c(dat$intercept,dat$beta))^2)
sum((RW_fit$post_mean$beta - c(dat$intercept,dat$beta))^2)
sum((glm_fit$coefficients - c(dat$intercept,dat$beta))^2)




HMC_fit10_e2 <- HMC_fit
HMC_fit20_e2 <- Bayes_logit_reg(dat$y,dat$X, method="HMC",
                           initial_beta, initial_sigma2_beta, include_intercept,
                           initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                           target_accept = 0.65, initial_step_size = 0.01,leapfrog_steps = 20)

HMC_fit30_e2 <- Bayes_logit_reg(dat$y,dat$X, method="HMC",
                             initial_beta, initial_sigma2_beta, include_intercept,
                             initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                             target_accept = 0.65, initial_step_size = 0.01,leapfrog_steps = 30)


HMC_fit20_e3 <- Bayes_logit_reg(dat$y,dat$X, method="HMC",
                             initial_beta, initial_sigma2_beta, include_intercept,
                             initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                             target_accept = 0.65, initial_step_size = 0.001,leapfrog_steps = 20)

HMC_fit50_e3 <- Bayes_logit_reg(dat$y,dat$X, method="HMC",
                                initial_beta, initial_sigma2_beta, include_intercept,
                                initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                                target_accept = 0.65, initial_step_size = 0.001,leapfrog_steps = 50)

HMC_fit80_e3 <- Bayes_logit_reg(dat$y,dat$X, method="HMC",
                                initial_beta, initial_sigma2_beta, include_intercept,
                                initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                                target_accept = 0.65, initial_step_size = 0.001,leapfrog_steps = 80)


HMC_fit20_e4 <- Bayes_logit_reg(dat$y,dat$X, method="HMC",
                             initial_beta, initial_sigma2_beta, include_intercept,
                             initial_intercept, mcmc_sample, burnin, thinning, step_adjust_accept, maxiter_adjust_accept,
                             target_accept = 0.65, initial_step_size = 0.0001,leapfrog_steps = 20)

logpost_ylim = c(-600,-50)
accept_ylim = c(0,1)

par(mfcol=c(1,1))
plot(RW_fit$trace$logpost,type="l",ylim=logpost_ylim)
lines(MALA_fit$trace$logpost,col="red")
lines(HMC_fit$trace$logpost,col="blue")
legend("bottomright",c("RW","MALA","HMC"),lty=1,col=c("black","red","blue"))

steps = 1:1000

par(mfrow=c(3,2))
plot(steps,HMC_fit10_e2$trace$logpost[steps],type="l",col="blue",
     main="HMC log density (epsilon = 1e-2)",ylim=logpost_ylim,ylab="")
lines(steps,HMC_fit20_e2$trace$logpost[steps],col="green")
lines(steps,HMC_fit30_e2$trace$logpost[steps],col="yellow")
legend("bottomright",c("L = 10","L = 20","L = 30"),lty=1,col=c("blue","green","yellow"))

plot(steps,HMC_fit10_e2$trace$accept_rate[steps],type="l",col="blue",
     main="HMC acceptance rate (epsilon = 1e-2)",ylim=accept_ylim,ylab="")
lines(steps,HMC_fit20_e2$trace$accept_rate[steps],col="green")
lines(steps,HMC_fit30_e2$trace$accept_rate[steps],col="yellow")
legend("bottomright",c("L = 10","L = 20","L = 30"),lty=1,col=c("blue","green","yellow"))

plot(steps,HMC_fit20_e2$trace$logpost[steps],type="l",col="blue",
     main="HMC log density (L = 20)",ylim=logpost_ylim,,ylab="")
lines(steps,HMC_fit20_e3$trace$logpost[steps],col="green")
lines(steps,HMC_fit20_e4$trace$logpost[steps],col="yellow")
legend("bottomright",c("epsilon = 1e-2","epsilon = 1e-3","epsilon = 1e-4"),
       lty=1,col=c("blue","green","yellow"))

plot(steps,HMC_fit20_e2$trace$accept_rate[steps],type="l",col="blue",
     main="HMC acceptance rate (L = 20)",ylim=accept_ylim,,ylab="")
lines(steps,HMC_fit20_e3$trace$accept_rate[steps],col="green")
lines(steps,HMC_fit20_e4$trace$accept_rate[steps],col="yellow")
legend("bottomright",c("epsilon = 1e-2","epsilon = 1e-3","epsilon = 1e-4"),lty=1,col=c("blue","green","yellow"))


plot(steps,HMC_fit20_e3$trace$logpost[steps],type="l",col="blue",
     main="HMC log density (epsilon = 1e-3)",ylim=logpost_ylim,,ylab="")
lines(steps,HMC_fit50_e3$trace$logpost[steps],col="green")
lines(steps,HMC_fit80_e3$trace$logpost[steps],col="yellow")
legend("bottomright",c("L = 20","L = 50","L = 80"),
       lty=1,col=c("blue","green","yellow"))

plot(steps,HMC_fit20_e3$trace$accept_rate[steps],type="l",col="blue",
     main="HMC acceptance rate (epsilon = 1e-3)",ylim=accept_ylim,ylab="")
lines(steps,HMC_fit50_e3$trace$accept_rate[steps],col="green")
lines(steps,HMC_fit80_e3$trace$accept_rate[steps],col="yellow")
legend("bottomright",c("L = 20","L = 50","L = 80"),lty=1,col=c("blue","green","yellow"))

