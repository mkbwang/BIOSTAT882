library(Rcpp)
library(MASS)

yvec = rnorm(100, 10, 1)

sourceCpp("HW1/P5.cpp")


result <- langevin5(yvec, chain_length=30000, sigma02=1e-8, 
                      init_mu1 = 5, init_mu2 = 5)

# acceptance
plot(result$acceptance, type='l',
     xlab = "Step", ylab="Average Rate", main="Average Acceptance Rate every 100 steps")

#mu_1
plot(x=25000:30000, y=result$mu1[25000:30000], type='l',xlab="Iteration", ylab="Estimated Value", main=expression(mu[1]))
hist(result$mu1[25000:30000], xlab="Value", ylab="Count", main=expression(mu[1]))
mean(result$mu1[25000:30000])
sd(result$mu1[25000:30000])

#mu_2
plot(x=25000:30000, y=result$mu2[25000:30000], type='l',xlab="Iteration", ylab="Estimated Value", main=expression(mu[2]))
hist(result$mu2[25000:30000], xlab="Value", ylab="Count", main=expression(mu[2]))
mean(result$mu2[25000:30000])
sd(result$mu2[25000:30000])
