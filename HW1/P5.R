library(Rcpp)
library(MASS)

yvec = rnorm(100, 10, 1)

sourceCpp("HW1/P5.cpp")


result <- langevin5(yvec, chain_length=40000, sigma02=1e-8, 
                      init_mu1 = 5, init_mu2 = 5, rw=FALSE)

# loglik
plot(x=1:40000, result$loglik[1:40000], type='l', xlab="Step", ylab="Log Posterior Likelihood")

# acceptance
plot(x=seq(1, 40000, 100), y=result$acceptance, type='l',
     xlab = "Step", ylab="Average Rate", main="Average Acceptance Rate every 100 steps")

#mu_1
plot(x=35000:40000, y=result$mu1[35000:40000], type='l',xlab="Iteration", ylab="Estimated Value", main=expression(mu[1]))
hist(result$mu1[35000:40000], xlab="Value", ylab="Count", main=expression(mu[1]))
mean(result$mu1[35000:40000])
sd(result$mu1[35000:40000])

#mu_2
plot(x=35000:40000, y=result$mu2[35000:40000], type='l',xlab="Iteration", ylab="Estimated Value", main=expression(mu[2]))
hist(result$mu2[25000:30000], xlab="Value", ylab="Count", main=expression(mu[2]))
mean(result$mu2[25000:30000])
sd(result$mu2[25000:30000])
