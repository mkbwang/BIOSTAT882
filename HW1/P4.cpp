#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix MH4(int run, int chain, double target, double proposal) {
  NumericMatrix allruns(run, chain+1); // store all the output values
  NumericVector currentstate = Rcpp::runif(run, target/2, target*2); // all runs start from 
  allruns(_, 0) = currentstate;
  for (int chainid=0; chainid < chain; chainid++){
    currentstate = allruns(_, chainid);// current state values for all runs
    NumericVector prop_values = Rcpp::rexp(run, proposal); // sample from proposal distribution
    NumericVector draw = Rcpp::runif(run); // sample from uniform distribution to decide accept/rejection
    // the log ratio threshold in the accept/rejection step
    NumericVector threshold = -target*prop_values - proposal*currentstate + proposal*prop_values + target*currentstate;
    allruns(_, chainid+1) = ifelse(log(draw) < threshold, prop_values, currentstate);
  }
  return allruns;
}



