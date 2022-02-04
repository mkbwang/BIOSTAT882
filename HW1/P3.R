library(Rcpp)

Rcpp::sourceCpp('HW1/P3.cpp')

simulation = as.vector(mc3(X0=0))

plot(simulation, type='l', xlab="Iteration", ylab="Value", main="Random Walk Markov Chain")

est_distribution = as.vector(mstep3(simulation, 4, 3))

hist(est_distribution)
