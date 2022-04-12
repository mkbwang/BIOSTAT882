#include <RcppDist.h>
#include <cmath>
#include <limits>
using namespace std;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

using namespace Rcpp;
using namespace arma;

// utility functions

// [[Rcpp::export]]
vec log1pexp_fast(vec& x){
  vec y = x;
  uvec ids = find(x <= -37.0);
  y.elem(ids) = exp(x.elem(ids));
  ids = find(x > -37.0 && x <= 18);
  y.elem(ids) = log1p(exp(x.elem(ids)));
  ids = find(x > 18.0 && x <= 33.3);
  y.elem(ids) = x.elem(ids) + exp(-x.elem(ids));
  return y;
}

// [[Rcpp::export]]
arma::vec probit(arma::vec& y, bool islower=true, bool takelog=false) {
  // y are link function values
  const int numindv = y.n_elem;
  arma::vec probs = arma::vec(numindv);
  for (int i = 0 ; i < numindv; i++){
    probs(i) = R::pnorm(y(i), 0, 1, islower, takelog);
  }
  return probs;
}

double kernel(double x1, double x2){
    double value = exp(- (x1-x2) * (x1-x2) / 2);
    return value;   
}

arma::vec gaussian(arma::vec& y){
  arma::vec density = arma::exp(- y % y / 2) / std::sqrt(2 * arma::datum::pi);
  return density;
}

arma::vec tnorm(arma::vec& x, arma::vec& y){
    // y are binary outcomes, x are the link function values
  const int numindv = y.n_elem;
  arma::vec z = arma::vec(numindv);
  for (int i=0; i < numindv; i++){
    if (y(i)==1){
      z(i) = r_truncnorm(x(i), 1, 0, numeric_limits<double>::max());
    } else{
      z(i) = r_truncnorm(x(i), 1, -numeric_limits<double>::max(), 0);
    }
  }
  return z;
}


// [[Rcpp::export]]
arma::vec mvrnormArma(arma::vec& mu, arma::mat& sigma) {
   int ncols = sigma.n_cols;
   arma::vec epsilon = arma::randn(ncols);
   return mu + arma::chol(sigma, "lower") * epsilon;
}

class BayesShrinkageGPReg{
    private:
        struct LinearRegData{
            int n; // number of observations
            int p; // number of covariates
            vec y; // binary outcome
            mat X; // covariate matrix
            cube K; // kernel matrices for each covariate
            cube V; // eigen vectors for each kernel matrix
            mat D; // eigen values for each kernel matrix
        } dat;

        struct HyperParas{
            double A; // prior half cauchy distribution for sigma
            double inv_A_sq;
            double prob; // prior bernoulli distribution 
        } hyperparas;

        struct LinearRegParas{
            vec delta; // indicator of covariates being included or not
            double inv_sigma_sq; //inverse of variance parameter
            double inv_a; // intermediate variable for half cauchy distribution of sigma2
            mat f_val; // function value for all the covariates
            double alpha; // common intercept
            vec z; // auxiliary assistive variable for probit distribution
        } paras;

        struct MCMCSample{
            mat delta;
            vec sigma_sq;
            vec alpha;
            cube f_val; 
        } paras_sample;

        struct GibbsSamplerProfile{
            vec loglik;
        } gibbs_profile;
  
        struct GibbsSamplerControl{
            int total_iter;
            int burnin;
            int mcmc_sample;
            int thinning;
            int verbose;
            int save_profile;
            int total_profile;
        } gibbs_control;

        int iter;

    public:

        void load_data(const vec& in_y, const mat& in_X){
            dat.y = in_y;
            dat.X = in_X;
            dat.p = dat.X.n_cols;
            dat.n = dat.X.n_rows;

            // calculate the kernel function for all the covariates
            dat.K.zeros(n, n, p);
            dat.V.zeros(n, n, p);
            dat.D.zeros(n, p);

            for (int i=0; i<p; i++){
                mat temp_K;
                temp_K.zeros(n, n);
                mat temp_V;
                temp_V.zeros(n, n);    
                vec temp_D;
                temp_D.zeros(n);

                for (int j=0; j<n; j++){ // off diagonal terms of K
                    double x1 = dat.X(j, i);
                    for (int k=j+1; k<n; k++){
                        double x2 = dat.X(k, i);
                        temp_K(j, k) = kernel(x1, x2);
                    }
                }

                temp_K = temp_K.t() + temp_K;
                for(int j=0; j<n; j++){ // diagonal terms of K
                    double x1 = dat.X(j, i);
                    temp_K(j, j) = kernel(x1, x1);
                }

                // eigen decomposition
                eig_sym(temp_D, temp_V, temp_K);

                dat.K.slice(i) = temp_K;
                dat.D.col(i) = temp_D;
                dat.V.slice(i) = temp_V;

            }
            
        };

        void set_hyperparas(const double& in_A, const double& in_pi){
            hyperparas.A = in_A;
            hyperparas.inv_A_sq = 1.0/(in_A * in_A);
            hyperparas.prob = in_pi;
        };

        void set_paras_initial_values(){
            
            paras.inv_a = randg(distr_param(1.0/2, 1.0/hyperparas.inv_A_sq));
            paras.inv_sigma_sq = randg(distr_param(1.0/2, 1.0/paras.inv_a));
            paras.alpha = randn() / sqrt(paras.inv_sigma_sq);
            vec intermediate = randu(dat.p);
            vec threshold = ones(dat.p) * hyperparas.prob;
            paras.delta = intermediate < threshold;
            vec prior_means = zeros(dat.n);

            for(int i=0; i<dat.p; i++){
                mat covar_mat = dat.K.slice(i) / sqrt(paras.inv_sigma_sq);
                paras.f_val.col(i) = mvrnormArma(prior_means, covar_mat);
            }

        };

        void set_gibbs_control(int in_mcmc_sample, int in_burnin, int in_thinning, 
                         int in_verbose, int in_save_profile){
            gibbs_control.mcmc_sample = in_mcmc_sample;
            gibbs_control.burnin = in_burnin;
            gibbs_control.thinning = in_thinning;
            gibbs_control.total_iter = gibbs_control.burnin;
            gibbs_control.total_iter += gibbs_control.mcmc_sample*gibbs_control.thinning; 
            gibbs_control.verbose = in_verbose;
            gibbs_control.save_profile = in_save_profile;
            if(gibbs_control.save_profile > 0){
            gibbs_control.total_profile = gibbs_control.total_iter/gibbs_control.save_profile;
            } else{
            gibbs_control.total_profile = 0;
            }
        };


        /*TODO: update parameters and likelihoods



        */


        // traces of parameters and likelihoods

        void initialize_paras_sample(){
            paras_sample.alpha.zeros(gibbs_control.mcmc_sample);
            paras_sample.delta.zeros(dat.p, gibbs_control.mcmc_sample);
            paras_sample.sigma_sq.zeros(gibbs_control.mcmc_sample);
            paras_sample.f_val.zeros(dat.n, dat.p, gibbs_control.mcmc_sample);
        };

        void save_paras_sample(){
            if(iter >= gibbs_control.burnin){
                if((iter - gibbs_control.burnin)%gibbs_control.thinning==0){
                    int mcmc_iter = (iter - gibbs_control.burnin)/gibbs_control.thinning;
                    paras_sample.alpha(mcmc_iter) = paras.alpha;
                    paras_sample.delta.col(mcmc_iter) = paras.delta;
                    paras_sample.sigma_sq(mcmc_iter) = 1.0/paras.inv_sigma_sq;
                    paras_sample.f_val.slice(mcmc_iter) = paras.f_val;
                }
            }
        };


        void initialize_gibbs_profile(){
            if(gibbs_control.save_profile>0){
                gibbs_profile.loglik.zeros(gibbs_control.total_profile);
            }
        };


        void save_gibbs_profile(){

        };

        void monitor_gibbs(){

        };

        void run_gibbs(){

        };

        // outputs

        List get_gibbs_post_mean(){

        };



        List get_gibbs_sample(){

        };


        List get_gibbs_trace(){

        };


        List get_gibbs_control(){
            return List::create(Named("total_iter") = gibbs_control.total_iter,
                        Named("burnin") = gibbs_control.burnin,
                        Named("mcmc_sample") = gibbs_control.mcmc_sample,
                        Named("thinning") = gibbs_control.thinning,
                        Named("verbose") = gibbs_control.verbose,
                        Named("save_profile") = gibbs_control.save_profile,
                        Named("total_profile") = gibbs_control.total_profile);
        }

        int get_iter(){
            return iter;
        };

}


//[[Rcpp::export]]
List simul_dat_linear(int n, double intercept, vec& beta, double X_rho, double X_sd,
                      double R_sq = 0.9){
  double sqrt_X_rho = sqrt(X_rho);
  mat X = X_sd*sqrt(1.0-X_rho)*randn<mat>(n,beta.n_elem);
  vec Z = X_sd*sqrt_X_rho*randn<vec>(n);
  for(int j=0; j<X.n_cols;j++){
    X.col(j) += Z;
  }
  
  vec mu = intercept + X*beta;
  double sigma_sq = (1 - R_sq)/R_sq*var(mu);
  vec epsilon = sqrt(sigma_sq)*randn<vec>(n);
  vec y = mu + epsilon;
  return List::create(Named("y") = y,
                      Named("X") = X,
                      Named("mu") = mu,
                      Named("intercept") = intercept,
                      Named("beta") = beta,
                      Named("X_rho") = X_rho,
                      Named("X_sd") = X_sd,
                      Named("R_sq") = R_sq,
                      Named("sigma_sq") = sigma_sq);
}

