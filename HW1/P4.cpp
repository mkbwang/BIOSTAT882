#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix MH4(int run, int chain, double target, double proposal) {
  NumericMatrix allruns(run, chain);
  NumericVector currentstate(run, 1.0);
  allruns(_, 0) = currentstate;
  for (int chainid=0; chainid < chain; chainid++){
    currentstate = allruns(_, chainid);
    NumericVector prop_values = Rcpp::rexp(run, proposal);
    NumericVector draw = Rcpp::runif(run);
    LogicalVector choice = draw < prop_values;
    for (int runid=0; runid < run; runid++){
      if (choice[runid]){
        allruns(runid, chainid+1) = prop_values[runid];
      } else{
        allruns(runid, chainid+1) = allruns(runid, chainid);
      }
    }
  }
  return allruns;
}



