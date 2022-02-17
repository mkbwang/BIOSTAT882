#include <Rcpp.h>
#include <iostream>
#include <math.h> 
using namespace Rcpp;


// [[Rcpp::plugins("cpp11")]]


// calculate the threshold for acceptance/rejection
NumericVector logthreshold(NumericVector Y, double mu1, double mu2, double mu1_prime, double mu2_prime, double sigma2, bool rw=false){
  NumericVector deviation = Y - mu1 - mu2;
  NumericVector deviation2 = deviation * deviation;
  double deviation_sum = sum(deviation);
  double deviation2_sum = sum(deviation2);
  NumericVector deviation_prime = Y - mu1_prime - mu2_prime;
  NumericVector deviation_prime2 = deviation_prime * deviation_prime;
  double deviation_prime_sum = sum(deviation_prime);
  double deviation_prime2_sum = sum(deviation_prime2);
  // posterior likelihood
  double logpostlik = -mu1*mu1/2 - 500*mu2*mu2 - 
    1/2*deviation2_sum;
  double logpostlik_prime = -mu1_prime*mu1_prime/2 - 500*mu2_prime*mu2_prime - 
    1/2*deviation_prime2_sum;
  
  // proposal distribution of mu given mu_prime
  double diff_mu1_mu1prime, diff_mu2_mu2prime;
  if (rw){
    diff_mu1_mu1prime = mu1 - mu1_prime ;
    diff_mu2_mu2prime = mu2 - mu2_prime ;
  } else{
    diff_mu1_mu1prime = mu1 - mu1_prime - sigma2 / 2 * 
      (-mu1_prime + deviation_prime_sum);
    diff_mu2_mu2prime = mu2 - mu2_prime - sigma2 / 2 * 
      (-1000*mu2_prime + deviation_prime_sum);
  }
  double log_proposal_mu_from_muprime = -1/(2*sigma2)*
    (diff_mu1_mu1prime * diff_mu1_mu1prime + diff_mu2_mu2prime * diff_mu2_mu2prime);
  
  // proposal distribution of mu_prime given mu
  double diff_mu1prime_mu1, diff_mu2prime_mu2;
  if (rw){
    diff_mu1prime_mu1 = mu1_prime - mu1 - sigma2 / 2 * 
      (-mu1 + deviation_sum);
    diff_mu2prime_mu2 = mu2_prime - mu2 - sigma2 / 2 * 
      (-1000*mu2 + deviation_sum);
  } else{
    diff_mu1prime_mu1 = mu1_prime - mu1;
    diff_mu2prime_mu2 = mu2_prime - mu2;
  }
  double log_proposal_muprime_from_mu = -1/(2*sigma2)*
    (diff_mu1prime_mu1 * diff_mu1prime_mu1 + diff_mu2prime_mu2 * diff_mu2prime_mu2);
  
  
  double threshold = logpostlik_prime + log_proposal_mu_from_muprime - 
    logpostlik - log_proposal_muprime_from_mu;
  
  return NumericVector::create(threshold, logpostlik, logpostlik_prime);
}


// [[Rcpp::export]]
List langevin5(NumericVector Y, int chain_length=10000, double sigma02=0.001, int period=100, double init_mu1=5, double init_mu2=5, bool rw=false) {
  NumericVector mu1(chain_length+1);
  NumericVector mu2(chain_length+1);
  NumericVector loglik(chain_length);
  int acceptance_change = chain_length / period;
  NumericVector acceptance(acceptance_change); // record acceptance rate change
  mu1[0] = init_mu1;
  mu2[0] = init_mu2;
  double sigma2 = sigma02;
  int A = 0;
  for (int i=0; i<chain_length; i++){
    // diffusion
    NumericVector Z=Rcpp::rnorm(2, 0, 1);
    double deviation_sum = sum(Y - mu1[i] - mu2[i]);
    double mu1_prime, mu2_prime;
    if (rw){
      mu1_prime = mu1[i] + Z[0] * std::sqrt(sigma2);
      mu2_prime = mu2[i] + Z[1] * std::sqrt(sigma2);
    } else{
      mu1_prime = mu1[i] + sigma2 / 2 * 
        (-mu1[i] + deviation_sum) +
        Z[0] * std::sqrt(sigma2);
      mu2_prime = mu2[i] + sigma2 / 2 * 
        (-1000*mu2[i] + deviation_sum)+
        Z[1] * std::sqrt(sigma2);
    }
    
    
    // accept or reject
    double U = R::runif(0, 1);
    NumericVector result = logthreshold(Y, mu1[i], mu2[i], mu1_prime, mu2_prime, sigma2);
    
    if (log(U)< result[0]){
      mu1[i+1] = mu1_prime;
      mu2[i+1] = mu2_prime;
      loglik[i] = result[2];
      A += 1;
    } else{
      mu1[i+1] = mu1[i];
      mu2[i+1] = mu2[i];
      loglik[i] = result[1];
    }
    
    if ( (i+1) % period == 0){//update sigma2
      acceptance[(i+1)/period-1] = A * 1.0 / period;
      double r = 1. + 1000. * pow((A*1.0/period - 0.574), 3);
      A = 0;
      if (r > 1.1){
        sigma2 = sigma2 * 1.1;
      } else if (r < 0.9){
        sigma2 = sigma2*0.9;
      } else{
        sigma2 = sigma2*r;
      }
    }
  }
  
  List result = List::create(Named("mu1") = mu1, 
                             Named("mu2") = mu2, 
                             Named("acceptance") =acceptance,
                             Named("loglik") = loglik);
  return result;
}



