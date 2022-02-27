#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::vec probit(arma::mat& X, arma::vec& y, bool islower=true, bool takelog=false) {
  const int numindv = X.n_rows;
  arma::vec XTy = X*y;
  arma::vec probs = vec(numindv);
  for (int i = 0 ; i < numindv; i++){
    probs(i, 0) = R::pnorm(XTy(i, 0), 0, 1, islower, takelog);
  }
  return probs;
}


