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

mcmc_sample = 1000

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

save(RW_fit, RW_duration, file='HW2/RW_fit.RData')

MALA_start <- proc.time()
MALA_fit <- Bayes_Probit_reg(mnist_train_y, mnist_train_x, method="MALA",
                             initial_beta, 
                             initial_sigma2_beta, include_intercept,
                             initial_intercept, mcmc_sample, burnin_MALA, thinning_MALA, step_adjust_accept,maxiter_adjust_accept_MALA,
                             target_accept = 0.5, initial_step_size = 1e-6, a_gamma = 0.01, b_gamma=0.01)
MALA_duration <- proc.time() - MALA_start



save(MALA_fit, MALA_duration, file='HW2/MALA_fit.RData')


HMC_start <- proc.time()
HMC_fit <- Bayes_Probit_reg(mnist_train_y, mnist_train_x, method="HMC",
                            initial_beta, 
                            initial_sigma2_beta, include_intercept,
                            initial_intercept, mcmc_sample, burnin_HMC, thinning_HMC, step_adjust_accept, maxiter_adjust_accept_HMC,
                            target_accept = 0.7, initial_step_size = 1e-4, a_gamma = 0.01, b_gamma=0.01)
HMC_duration <- proc.time() - HMC_start


save(HMC_fit, HMC_duration, file='HW2/HMC_fit.RData')

Gibbs_start <- proc.time()
Gibbs_fit <- Bayes_Probit_reg(mnist_train_y, mnist_train_x, method="Gibbs",
                              initial_beta, 
                              initial_sigma2_beta, include_intercept,
                              initial_intercept, mcmc_sample, burnin_Gibbs, thinning_Gibbs, step_adjust_accept, maxiter_adjust_accept_HMC,
                              target_accept = 0.7, initial_step_size = 1e-5, a_gamma = 0.01, b_gamma=0.01, verbose=100)
Gibbs_duration <- proc.time() - Gibbs_start

save(Gibbs_fit, Gibbs_duration, file='HW2/Gibbs_model.RData')


load('HW2/RW_fit.RData')
load('HW2/MALA_fit.RData')
load('HW2/HMC_fit.RData')
load('HW2/Gibbs_model.RData')

plot(RW_fit$trace$loglik, type='l', xlab="Iteration", ylab="Log Likelihood", main="Random Walk")
plot(RW_fit$trace$accept_rate, type='l', xlab="Iteration", ylab="Acceptance Rate", main="Random Walk")
RW_effsizes <- effectiveSize(t(RW_fit$mcmc$beta))
mean(RW_effsizes)
sd(RW_effsizes)


plot(MALA_fit$trace$loglik, type='l', xlab="Iteration", ylab="Log Likelihood", main="Metropolis Adjusted Langevin")
plot(MALA_fit$trace$accept_rate, type='l', xlab="Iteration", ylab="Acceptance Rate", main="Metropolis Adjusted Langevin")
MALA_effsizes <- effectiveSize(t(MALA_fit$mcmc$beta))
mean(MALA_effsizes)
sd(MALA_effsizes)

plot(HMC_fit$trace$loglik, type='l', xlab="Iteration", ylab="Log Likelihood", main="Hamiltonian Monte Carlo")
plot(HMC_fit$trace$accept_rate, type='l', xlab="Iteration", ylab="Acceptance Rate", main="Hamiltonian Monte Carlo")
HMC_effsizes <- effectiveSize(t(HMC_fit$mcmc$beta))
mean(HMC_effsizes)
sd(HMC_effsizes)

plot(Gibbs_fit$trace$loglik, type='l', xlab="Iteration", ylab="Log Likelihood", main="Gibbs Sampler")
plot(Gibbs_fit$trace$accept_rate, type='l', xlab="Iteration", ylab="Acceptance Rate", main="Gibbs Sampler")
Gibbs_effsizes <- effectiveSize(t(Gibbs_fit$mcmc$beta))
mean(Gibbs_effsizes)
sd(Gibbs_effsizes)



test_x <- read.csv('HW2/mnist_1000_test_X.csv') |> as.matrix()
test_y <- read.csv('HW2/mnist_1000_test_Y.csv') |>  as.matrix() |> as.vector()

test_x <- cbind(1, test_x) # add an intercept term

# somehow the first sample is zero, need to drop it
coefficients_RW <- RW_fit$mcmc$beta[, 2:1000] 
coefficients_MALA <- MALA_fit$mcmc$beta[, 2:1000]
coefficients_HMC <- HMC_fit$mcmc$beta[, 2:1000]
coefficients_Gibbs <- Gibbs_fit$mcmc$beta[, 2:1000]

RW_pred <- pnorm(test_x %*% coefficients_RW)
MALA_pred <- pnorm(test_x %*% coefficients_MALA)
HMC_pred <- pnorm(test_x %*% coefficients_HMC)
Gibbs_pred <- pnorm(test_x %*% coefficients_Gibbs)

RW_pred_avg <- rowMeans(RW_pred) 
MALA_pred_avg <- rowMeans(MALA_pred) 
HMC_pred_avg <- rowMeans(HMC_pred)
Gibbs_pred_avg <- rowMeans(Gibbs_pred) 

aggregate_pred_avg <- cbind(RW_pred_avg, MALA_pred_avg,
                            HMC_pred_avg, Gibbs_pred_avg)

# find the image with the largest prediction probability variance between MCMC methods
method_variance <- (aggregate_pred_avg - rowMeans(aggregate_pred_avg))^2 |> rowSums()
selected_id <- which.max(method_variance)
selected_image <- test_x[selected_id, 2:785]

# image function
plot_mnist <- function(img,layout = NULL){
  par(mar=c(0,0,0,0))
  if(!is.matrix(img)){
    
    image(1:28, 1:28, matrix(img, nrow=28)[ , 28:1], 
          col = gray(seq(0, 1, 0.05)), xlab = "", ylab="",axes = FALSE,asp=1)
  } else{
    n = nrow(img)
    m = round(sqrt(n))
    k = ceiling(n/m)
    if(is.null(layout)){
      layout = c(m,k)
    }
    par(mfrow=layout)
    for(i in 1:n){
      image(1:28, 1:28, matrix(img[i,], nrow=28)[ , 28:1], 
            col = gray(seq(0, 1, 0.05)), xlab = "", ylab="",axes = FALSE,asp=1)
    }
  }
  par(mfrow=c(1,1),mar=c(5, 4, 4, 2) + 0.1)
}
plot_mnist(selected_image)


# check out AUROC score
library(pROC)
RW_auc <- rep(0, 999)
for (i in 1:ncol(RW_pred)){
  RW_auc[i] <- auc(test_y, RW_pred[, i])
}
RW_result <- data.frame(AUC = RW_auc, Method = 'RW')

MALA_auc <- rep(0, 999)
for (i in 1:ncol(MALA_pred)){
  MALA_auc[i] <- auc(test_y, MALA_pred[, i])
}
MALA_result <- data.frame(AUC = MALA_auc, Method = 'MALA')

HMC_auc <- rep(0, 999)
for (i in 1:ncol(HMC_pred)){
  HMC_auc[i] <- auc(test_y, HMC_pred[, i])
}
HMC_result <- data.frame(AUC = HMC_auc, Method = 'HMC')

Gibbs_auc <- rep(0, 999)
for (i in 1:ncol(Gibbs_pred)){
  Gibbs_auc[i] <- auc(test_y, Gibbs_pred[, i])
}
Gibbs_result <- data.frame(AUC = Gibbs_auc, Method = 'Gibbs')

allresults <- rbind(RW_result, MALA_result, HMC_result, Gibbs_result)

library(ggplot2)
ggplot(allresults, aes(x=Method, y=AUC)) + geom_violin(draw_quantiles = c(.25, .50, .75)) + theme_bw(base_size=12) + 
  xlab('MCMC Method') + ylab('Area Under ROC Curve')
