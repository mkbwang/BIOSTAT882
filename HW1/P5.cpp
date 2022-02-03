#include <Rcpp.h>
#include <iostream>
#include <math.h> 
using namespace Rcpp;


// [[Rcpp::plugins("cpp11")]]


// calculate the threshold for acceptance/rejection
double logthreshold(NumericVector Y, double mu1, double mu2, double mu1_prime, double mu2_prime, double sigma2){
  NumericVector deviation = Y - mu1 - mu2;
  NumericVector deviation2 = deviation * deviation;
  NumericVector deviation_prime = Y - mu1_prime - mu2_prime;
  NumericVector deviation_prime2 = deviation_prime * deviation_prime;
  // posterior likelihood
  double loglikelihood = -mu1*mu1/2 - 500*mu2*mu2 - 
    1/2*std::accumulate(deviation2.begin(), deviation2.end(), 0);
  double loglikelihood_prime = -mu1_prime*mu1_prime/2 - 500*mu2_prime*mu2_prime - 
    1/2*std::accumulate(deviation_prime2.begin(), deviation_prime2.end(), 0);
  
  // proposal distribution of mu given mu_prime
  double diff_mu1_mu1prime = mu1 - mu1_prime + sigma2 / 2 * 
    (mu1_prime + std::accumulate(deviation_prime.begin(), deviation_prime.end(), 0));
  double diff_mu2_mu2prime = mu2 - mu2_prime + sigma2 / 2 * 
    (1000*mu2_prime + std::accumulate(deviation_prime.begin(), deviation_prime.end(), 0));
  double log_proposal_mu_from_muprime = -1/(2*sigma2)*
    (diff_mu1_mu1prime * diff_mu1_mu1prime + diff_mu2_mu2prime * diff_mu2_mu2prime);
  
  // proposal distribution of mu_prime given mu
  double diff_mu1prime_mu1 = mu1_prime - mu1 + sigma2 / 2 * 
    (mu1 + std::accumulate(deviation.begin(), deviation.end(), 0));
  double diff_mu2prime_mu2 = mu2_prime - mu2 + sigma2 / 2 * 
    (1000*mu2 + std::accumulate(deviation.begin(), deviation.end(), 0));
  double log_proposal_muprime_from_mu = -1/(2*sigma2)*
    (diff_mu1prime_mu1 * diff_mu1prime_mu1 + diff_mu2prime_mu2 * diff_mu2prime_mu2);
  
  
  double threshold = loglikelihood_prime + log_proposal_mu_from_muprime - 
    loglikelihood - log_proposal_muprime_from_mu;
  
  return threshold;
}


// [[Rcpp::export]]
List langevin5(NumericVector Y, int chain_length=10000, double sigma02=0.001, int period=100, double init_mu1=5, double init_mu2=5) {
  NumericVector mu1(chain_length+1);
  NumericVector mu2(chain_length+1);
  int acceptance_change = chain_length / period;
  NumericVector acceptance(acceptance_change); // record acceptance rate change
  mu1[0] = init_mu1;
  mu2[0] = init_mu2;
  double sigma2 = sigma02;
  int A = 0;
  for (int i=0; i<chain_length; i++){
    // diffusion
    NumericVector Z=Rcpp::rnorm(2, 0, std::sqrt(sigma2));
    NumericVector deviation = Y - mu1[i] - mu2[i];
    double mu1_prime = mu1[i] - sigma2 / 2 * 
      (mu1[i] + std::accumulate(deviation.begin(), deviation.end(), 0)) +
      Z[0];
    double mu2_prime = mu2[i] - sigma2 / 2 * 
      (1000*mu2[i] + std::accumulate(deviation.begin(), deviation.end(), 0))+
      Z[1];
    
    // accept or reject
    double U = R::runif(0, 1);
    double threshold = logthreshold(Y, mu1[i], mu2[i], mu1_prime, mu2_prime, sigma2);
    
    if (log(U)< threshold){
      mu1[i+1] = mu1_prime;
      mu2[i+1] = mu2_prime;
      A += 1;
    } else{
      mu1[i+1] = mu1[i];
      mu2[i+1] = mu2[i];
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
                             Named("acceptance") =acceptance);
  return result;
}



