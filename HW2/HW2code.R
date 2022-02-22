library(Rcpp)
library(RcppArmadillo)

sourceCpp('HW2/probittrial.cpp')

Xmat = as.matrix(cbind(c(0.1, 0.2, 0.3), c(-0.2,-0.3,0.1), c(0.1, -0.1, 0.2)))
yvec = c(1,2,1)

result = probit(Xmat, yvec)

pnorm(Xmat %*% yvec)
