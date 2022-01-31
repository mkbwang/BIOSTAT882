library(Rcpp)

Rcpp::sourceCpp('HW1/P4.cpp')

newmat = as.matrix(MH4(4, 10000, 1, 0.1))
