library(Rcpp)
library(MASS)

Rcpp::sourceCpp('HW1/P4.cpp')

mat1 = as.matrix(MH4(50, 10000, 1, 0.1))
est_params1 = rep(0, 50)
for (j in 1:50){
  est_params1[j] = fitdistr(mat1[j, 9001:10000], 'exponential')$estimate
}

mat2 = as.matrix(MH4(50, 10000, 1, 5))
est_params2 = rep(0, 50)
for (j in 1:50){
  est_params2[j] = fitdistr(mat2[j, 9001:10000], 'exponential')$estimate
}

comparison <- data.frame(Estimate = c(est_params1, est_params2),
                         Proposal = rep(c(0.1, 5), each=50))
comparison$Proposal <- as.factor(comparison$Proposal)

library(ggplot2)
ggplot(comparison, aes(x=Proposal, y=Estimate)) + geom_boxplot() + 
  xlab("Rate of Proposal Exponential Distribution") + ylab("Values") + 
  ggtitle("Posterior Estimate of Exponential Distribution Rate")
