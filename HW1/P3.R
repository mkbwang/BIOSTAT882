library(Rcpp)

Rcpp::sourceCpp('HW1/P3.cpp')

simulation = as.vector(mc3(X0=0))


est_distribution = as.vector(mstep3(simulation, 4, 3))

hist(est_distribution)
