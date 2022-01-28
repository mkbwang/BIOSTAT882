#include <Rcpp.h>

using namespace Rcpp;


// refer to https://teuder.github.io/rcpp4everyone_en/ for tips

// [[Rcpp::export]]
NumericVector mc3(double X0, double deltaP=0.5, double itaP=0.5, double normSD=1) {
  NumericVector output(10001);
  double X=X0;
  output[0] = X;
  double delta, ita, epsilon;
  for (int index=1; index<=10000; index++){
    delta = R::rbinom(1, deltaP);
    ita = 2 * R::rbinom(1, itaP) - 1;
    epsilon = R::rnorm(0, normSD);
    X=X+delta*ita + (1-delta)*epsilon;
    output[index] = X;
  }
  return output;  
}

// [[Rcpp::export]]
NumericVector mstep3(NumericVector chain, double x, int step){
  int clength = chain.length();
  int num_samples = clength - step; // number of sample points
  NumericVector output(num_samples, x);
  for (int index=0; index<num_samples; index++){
    output[index] += chain[index+step] - chain[index];
  }
  return output;
}


