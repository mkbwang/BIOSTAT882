#include <RcppDist.h>
#include <cmath>
#include <limits>
using namespace std;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

using namespace Rcpp;
using namespace arma;

// utility functions


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

double kernel(double x1, double x2, double ell=500){
    // simple matern kernel
    double value = exp(-abs(x1-x2) / ell);
    return value;   
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
            double loglik; // likelihood
            double logpost; // log posterior likelihood
        } paras;

        struct MCMCSample{
            mat delta;
            vec sigma_sq;
            vec alpha;
            cube f_val; 
        } paras_sample;

        struct GibbsSamplerProfile{
            vec loglik;
            vec logpost;
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
            dat.K.zeros(dat.n, dat.n, dat.p);
            dat.V.zeros(dat.n, dat.n, dat.p);
            dat.D.zeros(dat.n, dat.p);

            for (int i=0; i<dat.p; i++){
                mat temp_K;
                temp_K.zeros(dat.n, dat.n);
                mat temp_V;
                temp_V.zeros(dat.n, dat.n);    
                vec temp_D;
                temp_D.zeros(dat.n);

                for (int j=0; j<dat.n; j++){ // off diagonal terms of K
                    double x1 = dat.X(j, i);
                    for (int k=j+1; k<dat.n; k++){
                        double x2 = dat.X(k, i);
                        temp_K(j, k) = kernel(x1, x2);
                    }
                }

                temp_K = temp_K.t() + temp_K;
                for(int j=0; j<dat.n; j++){ // diagonal terms of K
                    temp_K(j, j) = 1;
                }

                // eigen decomposition
                bool success = eig_sym(temp_D, temp_V, temp_K);
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
            paras.delta = conv_to<vec>::from(intermediate < ones(dat.p)*hyperparas.prob );
            vec prior_means = zeros(dat.n);

            paras.f_val.zeros(dat.n, dat.p);    
            for(int i=0; i<dat.p; i++){
                mat covar_mat = dat.K.slice(i) / paras.inv_sigma_sq;
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



        void update_z(){
            vec link = paras.f_val * paras.delta + paras.alpha; // link function value
            paras.z = tnorm(link, dat.y);                
        };

        void update_selected_f(){
            // update the function values for covariates that are currently selected
            for (int i=0; i<dat.p; i++){
                if (paras.delta(i) == 0){
                    continue; // function not selected doesn't need to change here
                }
                vec D_i = dat.D.col(i);
                vec inv_D_i = 1.0 / D_i;
                vec intermediate = 1.0 / sqrt(paras.inv_sigma_sq * inv_D_i + 1.0);
                vec link = paras.f_val * paras.delta + paras.alpha;
                vec observed_f_i = paras.z - link + paras.f_val.col(i);
                vec rho = randn(size(observed_f_i));
                paras.f_val.col(i) = dat.V.slice(i) * (intermediate % (rho + intermediate % observed_f_i));
            }
        }
        
        void update_alpha(){
            // normal normal conjugate
            vec w = paras.z - paras.f_val * paras.delta;
            double var_alpha = 1.0/(dat.n + paras.inv_sigma_sq);
            double mean_alpha = var_alpha * accu(w);
            paras.alpha = mean_alpha + randn() * sqrt(var_alpha);
        }

        void update_inv_sigma_sq(){
            // normal inverse gamma conjugate
            double gamma_a = 0.5; // prior parameter for gamma distribution
            double gamma_b = paras.inv_a; // prior parameter for gamma distribution
            // first add the contribution of alpha
            gamma_a += 0.5;
            gamma_b += paras.alpha * paras.alpha / 2;
            // then add the contribution of all the covariates that are currently included
            for (int i=0; i<dat.p; i++){
                if (paras.delta(i) == 0){
                    continue;
                }
                gamma_a += dat.n / 2.0;
                vec D_i = dat.D.col(i);
                vec inv_sqrt_D_i = 1.0 / sqrt(D_i);
                mat V_i = dat.V.slice(i);
                vec intermediate = inv_sqrt_D_i % (V_i.t() * paras.f_val.col(i));
                gamma_b += accu(intermediate % intermediate) / 2.0;
            }
            paras.inv_sigma_sq = randg(distr_param(gamma_a, 1.0/gamma_b));
        }

        void update_unselected_f(){
            // update the function values for covariates that are currently unselected
            // based on gaussian process with the new sigma_sq
            for (int i=0; i<dat.p; i++){
                if (paras.delta(i) == 1){
                    continue;
                }
                vec prior_mean = zeros(dat.n);
                mat covar_mat = dat.K.slice(i) / paras.inv_sigma_sq;
                paras.f_val.col(i) = mvrnormArma(prior_mean, covar_mat);
            }
        }

        void update_inv_a(){
            // update the auxiliary variable a for half cauchy distribution
            double new_gamma_a = 1;
            double new_gamma_b = paras.inv_sigma_sq + hyperparas.inv_A_sq;
            paras.inv_a = randg(distr_param(new_gamma_a, 1.0/new_gamma_b));
        }

        void update_delta(){
            // update the deltas based on posterior binomial distribution
            for (int i=0; i<dat.p; i++){
                vec temp_delta = paras.delta;
                // get the probability when the pixel of interest is not selected
                temp_delta(i) = 0;
                vec link_off = paras.f_val * temp_delta + paras.alpha;
                vec probs_off = probit(link_off); // probabilities
                vec liks_off = abs(1.0 - dat.y - probs_off);
                // get the probability when the pixel of interest is selected
                temp_delta(i) = 1;
                vec link_on = paras.f_val * temp_delta + paras.alpha;
                vec probs_on = probit(link_on); // probabilities
                vec liks_on = abs(1.0 - dat.y - probs_on);

                double post_logit = accu(log(liks_on)) + log(hyperparas.prob) - log(1.0 - hyperparas.prob) -
                        accu(log(liks_off));
                
                double post_prob = 1.0 / (1.0 + exp(-post_logit));
                paras.delta(i) = (randu() < post_prob)? 1 : 0;
            }
        }

        void update_loglik(){
            vec link = paras.f_val * paras.delta + paras.alpha;
            vec probs = probit(link); // probability for all observations
            vec liks = abs(1 - dat.y - probs);
            paras.loglik = accu(log(liks));
        }

        void update_logpost(){
            paras.logpost = paras.loglik;
            double gpkernel = 0;
            for (int i=0; i<dat.p; i++){
                vec D_i = dat.D.col(i);
                vec inv_sqrt_D_i = 1.0 / sqrt(D_i);
                mat V_i = dat.V.slice(i);
                vec intermediate = inv_sqrt_D_i % (V_i.t() * paras.f_val.col(i));
                gpkernel -= accu(intermediate % intermediate) / 2;
            }
            // log prior of all the function values(GP)
            paras.logpost += gpkernel + dat.n * dat.p / 2 * log(paras.inv_sigma_sq);
            // log prior of all the deltas(bernoulli)
            paras.logpost += accu(log(abs(1.0 - paras.delta - hyperparas.prob)));
            // log prior of alpha(normal)
            paras.logpost += 0.5 * log(paras.inv_sigma_sq) - paras.inv_sigma_sq * paras.alpha * paras.alpha / 2;
            // log prior of inv sigma square(half cauchy, aka multilevel inverse gamma)
            paras.logpost += -0.5 * log(paras.inv_sigma_sq) - paras.inv_a * (paras.inv_sigma_sq + hyperparas.inv_A_sq);
        }

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
                gibbs_profile.logpost.zeros(gibbs_control.total_profile);
            }
        };


        void save_gibbs_profile(){
            if(gibbs_control.save_profile > 0){
                if(iter%gibbs_control.save_profile==0){
                    int profile_iter = iter/gibbs_control.save_profile;
                    update_loglik();
                    update_logpost();
                    gibbs_profile.loglik(profile_iter) = paras.loglik;
                    gibbs_profile.logpost(profile_iter) = paras.logpost;
                }
            }
        };

        void monitor_gibbs(){
            if(gibbs_control.verbose > 0){
                if(iter%gibbs_control.verbose==0){
                    std::cout << "iter: " << iter << " loglik: "<< paras.loglik << "logpost: " << paras.logpost <<   std::endl;
                }
            }
        };

        void run_gibbs(){
            initialize_paras_sample();
            initialize_gibbs_profile();
            for(iter=0; iter<gibbs_control.total_iter;iter++){
                update_z();
                update_selected_f();
                update_alpha();
                update_inv_sigma_sq();
                update_unselected_f();
                update_inv_a();
                update_delta();
                save_paras_sample();
                save_gibbs_profile();
                monitor_gibbs();
            }
        };

        // outputs

        List get_gibbs_post_mean(){
            List output;
            vec delta_mean = mean(paras_sample.delta, 1);
            output = List::create(Named("delta") = delta_mean,
                        Named("sigma_sq") = mean(paras_sample.sigma_sq),
                        Named("alpha") = mean(paras_sample.alpha));
            return output;
        };



        List get_gibbs_sample(){
            List output;
            output = List::create(Named("delta") = paras_sample.delta,
                          Named("sigma_sq") = paras_sample.sigma_sq,
                          Named("alpha") = paras_sample.alpha,
                          Named("f_val") = paras_sample.f_val);
            return output;
        };


        List get_gibbs_trace(){
            uvec iters = linspace<uvec>(1,gibbs_control.total_iter,gibbs_control.total_profile);
            return List::create(Named("iters") = iters,
                        Named("loglik") = gibbs_profile.loglik,
                        Named("logpost") = gibbs_profile.logpost);
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

};


//[[Rcpp::export]]
List simul_dat_Probit(int n, double intercept, vec& beta, double X_rho, double X_sd){
  double sqrt_X_rho = sqrt(X_rho);
  mat X = X_sd*sqrt(1.0-X_rho)*randn<mat>(n,beta.n_elem);
  vec Z = X_sd*sqrt_X_rho*randn<vec>(n);
  for(int j=0; j<X.n_cols;j++){
    X.col(j) += Z;
  }
  vec link = intercept + X*beta;
  vec prob = probit(link, true, false);
  vec u = randu<vec>(n);
  vec y = conv_to<vec>::from(u < prob);
  double R2 = var(prob)/var(y);
  return List::create(Named("y") = y,
                      Named("X") = X,
                      Named("prob") = prob,
                      Named("intercept") = intercept,
                      Named("beta") = beta,
                      Named("X_rho") = X_rho,
                      Named("X_sd") = X_sd,
                      Named("R2") = R2);
};


//[[Rcpp::export]]
List Bayes_shrinkage_GP_reg(
    vec& y, mat& X, 
    double prior_delta_prob=0.5,
    double A = 1, 
    int mcmc_sample = 500, 
    int burnin = 5000, 
    int thinning = 10,
    int verbose = 100,
    int save_profile = 1){

    wall_clock timer;
    timer.tic();

    BayesShrinkageGPReg model;
    model.load_data(y, X);
    model.set_hyperparas(A, prior_delta_prob);
    model.set_gibbs_control(mcmc_sample,
                            burnin,
                            thinning,
                            verbose,
                            save_profile);
    
    model.set_paras_initial_values();
    model.run_gibbs(); 

    double elapsed = timer.toc();

    List output;
    output = List::create(Named("post_mean") = model.get_gibbs_post_mean(),
                          Named("mcmc") = model.get_gibbs_sample(),
                          Named("trace") = model.get_gibbs_trace(),
                          Named("mcmc_control") = model.get_gibbs_control(),
                          Named("elapsed") = elapsed);
    
    return output;

};

//[[Rcpp::export]]
arma::mat GPpredict(mat& train_x, mat& sample_f_x, mat& test_x, double sigma2){

    const int train_n = train_x.n_rows;
    const int test_n = test_x.n_rows;
    const int num_feature = train_x.n_cols;

    mat predicted_f = zeros(test_n, num_feature);

    for (int i=0; i<num_feature; i++){
        mat kernel_test_train = zeros(test_n, train_n);
        mat kernel_train_train = zeros(train_n, train_n);
        mat kernel_test_test = zeros(test_n, test_n);
        vec train_vec = train_x.col(i);
        vec test_vec = test_x.col(i);
        vec train_f = sample_f_x.col(i);

        // calculate the kernel for training data
        for (int j=0; j<train_n; j++){
            double x1 = train_vec(j);
            for (int k=j+1; k<train_n; k++){
                double x2 = train_vec(k);
                kernel_train_train(j, k) = kernel(x1, x2);
            } 
        }
        kernel_train_train = kernel_train_train.t() + kernel_train_train;
        for (int j=0; j<train_n; j++){
            kernel_train_train(j, j) = 1;
        }

        mat inv_kernel_train_train = inv_sympd(kernel_train_train);
        //std::cout << "train kernel " << i << ": " << inv_kernel_train_train.is_sympd() << std::endl;


        // calculate the kernel for test data
        for (int j=0; j<test_n; j++){
            double x1 = test_vec(j);
            for (int k=j+1; k<test_n; k++){
                double x2 = test_vec(k);
                kernel_test_test(j, k) = kernel(x1, x2);
            }
        } 
        kernel_test_test = kernel_test_test.t() + kernel_test_test;
        for (int j=0; j<test_n; j++){
            kernel_test_test(j, j) = 1;
        }
        //std::cout << "test kernel " << i << ": " << kernel_test_test.is_sympd() << std::endl;

        // calculate the kernel from test data to training data
        for (int j=0; j<test_n; j++){
            double x1 = test_vec(j);
            for (int k=0; k<train_n; k++){
                double x2 = train_vec(k);
                kernel_test_train(j, k) = kernel(x1, x2);
            }
        }

        vec pred_mean = kernel_test_train * inv_kernel_train_train * train_vec;
        // numerical issues with matrix times transpose of matrix
        mat intermediate = kernel_test_train * inv_kernel_train_train *  kernel_test_train.t();
        mat intermediate_avoid_error = (intermediate + intermediate.t()) / 2.0;
        
        mat pred_var = kernel_test_test - intermediate_avoid_error;
        

        predicted_f.col(i) = mvrnormArma(pred_mean, pred_var);

    }

    return predicted_f;
    
};