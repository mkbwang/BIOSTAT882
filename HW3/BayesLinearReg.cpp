#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

class BayesLinearReg{
private:
  int method;//0: Gibbs sampler; 1: Mean field variational inference
  struct LinearRegData{
    int num_subjects;
    int num_predictors;
    vec y;
    mat X;
    vec Xty;
    mat XtX;
    double sum_y_sq;
  } dat;
  
  // struct HyperParas{
  //   double sigma_sq_beta;
  //   double inv_sigma_sq_beta;
  //   double A;
  //   double inv_A_sq;
  // } hyperparas;
  
  struct LinearRegParas{
    vec beta;
    double inv_sigma_sq;
    double inv_sigma_beta_sq;
    double inv_a_beta;
    double SSE;// ||y - X*beta||^2
    double beta_innerprod;// t(beta)*beta
    double loglik; 
    double logpost;
  } paras;
  
  struct MCMCSample{
    mat beta;
    vec sigma_sq;
    vec sigma_beta_sq;
    vec a_beta;
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
  
  
  struct VBLinearRegParas{
    vec E_beta;
    mat Cov_beta;
    double E_inv_sigma_sq;
    double E_inv_sigma_beta_sq;
    double E_inv_a_beta;
    double E_Xbeta_sq;
    double E_SSE;
    double E_beta_innerproduct;
    double ELBO;
  } vb_paras;
  
  struct VBProfile{
    vec ELBO;
  } vb_profile;
  
  struct VBControl{
    int max_iter;
    double para_diff_tol;
    int ELBO_stop;
    double ELBO_diff_tol;
    int verbose;
    int save_profile;
    int total_profile;
  } vb_control;
  
  int iter;
  
public:
  void set_method(CharacterVector in_method){
    if(in_method(0)=="Gibbs"){
      std::cout << "Gibbs Sampler" << std::endl;
      method = 0;
    } else if(in_method(0)=="MFVB"){
      std::cout << "Mean Field Variational Bayes" << std::endl;
      method = 1;
    } 
  };
  
  int get_method(){
    return method;
  }
  
  void load_data(const vec& in_y, const mat& in_X){
    dat.y = in_y;
    dat.X = in_X;
    dat.XtX = dat.X.t()*dat.X;
    dat.Xty = dat.X.t()*dat.y;
    dat.sum_y_sq = dot(dat.y,dat.y);
    dat.num_predictors = dat.X.n_cols;
    dat.num_subjects = dat.X.n_rows;
  };
  
  // void set_hyperparas(const double& in_sigma_sq_beta, const double& in_A){
  //   hyperparas.sigma_sq_beta = in_sigma_sq_beta;
  //   hyperparas.A = in_A;
  //   hyperparas.inv_sigma_sq_beta = 1.0/hyperparas.sigma_sq_beta;
  //   hyperparas.inv_A_sq = 1.0/(hyperparas.A*hyperparas.A);
  // };
  
  void set_paras_initial_values(const double& in_sigma_sq, const double& in_sigma_beta_sq){
    if(method==0){
      paras.beta = zeros(dat.num_predictors);
      paras.inv_sigma_sq = 1.0/in_sigma_sq;
      paras.inv_sigma_beta_sq = 1.0/in_sigma_beta_sq;
      update_beta();
      update_SSE();
      update_beta_innerprod();
      update_inv_sigma_sq();
      update_inv_a_beta();
      update_inv_sigma_beta_sq();
    } else if (method==1){
      vb_paras.Cov_beta = zeros(dat.num_predictors,dat.num_predictors);
      vb_paras.E_beta = zeros(dat.num_predictors);
      vb_paras.E_inv_sigma_sq = 1.0/in_sigma_sq;
      vb_paras.E_inv_sigma_beta_sq = 1.0/in_sigma_beta_sq;
      update_Cov_beta();
      update_E_beta();
      update_E_inv_a_beta();
      update_E_Xbeta_sq();
      update_E_SSE();
      update_E_beta_innerproduct();
      update_E_inv_sigma_sq();
      update_E_inv_sigma_beta_sq();
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
  
  void update_beta(){
    //std::cout << "inv_sigma_sq = " << paras.inv_sigma_sq << std::endl;
    mat Omega = dat.XtX*paras.inv_sigma_sq;
    Omega.diag() += paras.inv_sigma_beta_sq;
    //std::cout <<  Omega << std::endl;
    mat R = chol(Omega);
    
    vec temp = solve(R.t(),dat.Xty*paras.inv_sigma_sq);
    
    vec z = randn<vec>(dat.num_predictors);
    z += temp;
    
    paras.beta = solve(R,z);
  };
  
  void update_SSE(){
    vec residuals = dat.y - dat.X*paras.beta;
    paras.SSE = dot(residuals,residuals);
  }
  
  void update_beta_innerprod(){
    paras.beta_innerprod = dot(paras.beta,paras.beta);
  }

  void update_inv_sigma_sq(){
    paras.inv_sigma_sq = randg(distr_param(dat.num_subjects*0.5, 1.0/(0.5*paras.SSE)));
  };
  
  void update_inv_a_beta(){
    paras.inv_a_beta = randg(distr_param(1.0, 1.0/(paras.inv_sigma_beta_sq + 1)));
  };
  
  void update_inv_sigma_beta_sq(){
    paras.inv_sigma_beta_sq = randg(distr_param((dat.num_predictors+1.0)/2, 1.0/(0.5*paras.beta_innerprod +paras.inv_a_beta) ));
  }

  void update_loglik(){
    paras.loglik = dat.num_subjects*log(paras.inv_sigma_sq) - paras.inv_sigma_sq*paras.SSE;
    paras.loglik *=0.5;
  }
  
  void update_logpost(){
    paras.logpost = 0.5*(dat.num_predictors*log(paras.inv_sigma_beta_sq) - paras.inv_sigma_beta_sq*paras.beta_innerprod);
    paras.logpost += 1.5*log(paras.inv_sigma_beta_sq);
    paras.logpost += -paras.inv_a_beta*(paras.inv_sigma_beta_sq+1) + 2*log(paras.inv_a_beta) + log(paras.inv_sigma_sq);
    paras.logpost += paras.loglik;
  }
  
  void initialize_paras_sample(){
    paras_sample.beta.zeros(paras.beta.n_elem,gibbs_control.mcmc_sample);
    paras_sample.sigma_sq.zeros(gibbs_control.mcmc_sample);
    paras_sample.sigma_beta_sq.zeros(gibbs_control.mcmc_sample);
    paras_sample.a_beta.zeros(gibbs_control.mcmc_sample);
  }
  
  void save_paras_sample(){
    if(iter >= gibbs_control.burnin){
      if((iter - gibbs_control.burnin)%gibbs_control.thinning==0){
        int mcmc_iter = (iter - gibbs_control.burnin)/gibbs_control.thinning;
        paras_sample.beta.col(mcmc_iter) = paras.beta;
        paras_sample.sigma_sq(mcmc_iter) = 1.0/paras.inv_sigma_sq;
        paras_sample.sigma_beta_sq(mcmc_iter) = 1.0/paras.inv_sigma_beta_sq;
        paras_sample.a_beta(mcmc_iter) = 1.0/paras.inv_a_beta;
      }
    }
  };
  
  void initialize_gibbs_profile(){
    //std::cout << "total_profile: " << gibbs_control.total_profile << std::endl;
    if(gibbs_control.save_profile>0){
      gibbs_profile.loglik.zeros(gibbs_control.total_profile);
      gibbs_profile.logpost.zeros(gibbs_control.total_profile);
    }
  }
  
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
  }
  
  void monitor_gibbs(){
    if(gibbs_control.verbose > 0){
      if(iter%gibbs_control.verbose==0){
        std::cout << "iter: " << iter << " inv_sigma_sq: "<< paras.inv_sigma_sq  << " loglik: "<< paras.loglik <<  " logpost: "<< paras.logpost << std::endl;
      }
    }
  }
  
  void run_gibbs(){
    initialize_paras_sample();
    initialize_gibbs_profile();
    //std::cout << "total iter:" << gibbs_control.total_iter << std::endl;
    for(iter=0; iter<gibbs_control.total_iter;iter++){
      update_beta();
      update_SSE();
      update_beta_innerprod();
      update_inv_sigma_sq();
      update_inv_a_beta();
      update_inv_sigma_beta_sq();
      save_paras_sample();
      //std::cout << "save paras:" << paras_sample.beta.n_cols << std::endl;
      save_gibbs_profile();
      monitor_gibbs();
    }
  };
  
  void set_vb_control(int in_max_iter, 
                      double in_para_diff_tol, 
                      int in_ELBO_stop,
                      double in_ELBO_diff_tol,
                      int in_verbose,
                      int in_save_profile){
    vb_control.max_iter = in_max_iter;
    vb_control.para_diff_tol = in_para_diff_tol;
    vb_control.ELBO_stop = in_ELBO_stop;
    vb_control.ELBO_diff_tol = in_ELBO_diff_tol;
    vb_control.verbose = in_verbose;
    vb_control.save_profile = in_save_profile;
    if(vb_control.save_profile > 0){
      vb_control.total_profile = vb_control.max_iter/vb_control.save_profile;
    } else{
      vb_control.total_profile = 0;
    }
  };
  
  void update_Cov_beta(){
    mat temp = vb_paras.E_inv_sigma_sq*dat.XtX;
    temp.diag() += vb_paras.E_inv_sigma_beta_sq;
    vb_paras.Cov_beta = inv(temp);
  };
  
  void update_E_beta(){
    vb_paras.E_beta = vb_paras.Cov_beta*dat.Xty;
    vb_paras.E_beta *= vb_paras.E_inv_sigma_sq;
  };
  
  void update_E_inv_a_beta(){
    vb_paras.E_inv_a_beta = 1.0/(vb_paras.E_inv_sigma_beta_sq + 1);
  };
  
  void update_E_Xbeta_sq(){
    mat temp = dat.XtX*(vb_paras.Cov_beta + vb_paras.E_beta*vb_paras.E_beta.t());
    vb_paras.E_Xbeta_sq = trace(temp);
  }

  void update_E_SSE(){
    vb_paras.E_SSE = dat.sum_y_sq - 2*dot(vb_paras.E_beta, dat.Xty) + vb_paras.E_Xbeta_sq;
  };

  void update_E_beta_innerproduct(){
    vb_paras.E_beta_innerproduct = dot(vb_paras.E_beta, vb_paras.E_beta) + trace(vb_paras.Cov_beta);
  }

  void update_E_inv_sigma_sq(){
    vb_paras.E_inv_sigma_sq = 1.0*dat.num_subjects/vb_paras.E_SSE;
  };
  
  void update_E_inv_sigma_beta_sq(){
    vb_paras.E_inv_sigma_beta_sq = (dat.num_predictors+1.0)/(vb_paras.E_beta_innerproduct + 2*vb_paras.E_inv_a_beta);
  }


  void update_ELBO(){
    vb_paras.ELBO = -vb_paras.E_inv_sigma_sq * dot(vb_paras.E_beta, dat.Xty);
    vb_paras.ELBO += 0.5 * vb_paras.E_inv_sigma_sq *vb_paras.E_Xbeta_sq;
    vb_paras.ELBO += 0.5 * vb_paras.E_inv_sigma_beta_sq * vb_paras.E_beta_innerproduct;
    vb_paras.ELBO += vb_paras.E_inv_sigma_beta_sq * vb_paras.E_inv_a_beta;
  };
  
  double compute_paras_diff(vec& beta, vec& beta_prev){
    vec temp = beta - beta_prev;
    return dot(temp, temp)/beta.n_elem;
  };
  
  void initialize_vb_profile(){
    if(vb_control.save_profile>0){
      vb_profile.ELBO = zeros(vb_control.total_profile);
    }
  }
  
  void save_vb_profile(){
    if(vb_control.save_profile > 0){
      if(iter%vb_control.save_profile==0){
        int profile_iter = iter/vb_control.save_profile;
        if(vb_control.ELBO_stop==0){
          update_ELBO();
        }
        vb_profile.ELBO(profile_iter) = vb_paras.ELBO;
      }
    }
  }
  
  void monitor_vb(){
    if(vb_control.verbose > 0){
      if(iter%vb_control.verbose==0){
        if(vb_control.ELBO_stop==0){
          update_ELBO();
        }
        // std::cout << "iter: " << iter <<  " ELBO: "<< vb_paras.ELBO << std::endl;
      }
    }
  }
  
  void run_mfvb(){
    initialize_vb_profile();
    for(iter=0; iter<vb_control.max_iter; iter++){
      // std::cout << "Iteration " <<  iter << std::endl;
      vec E_beta_prev = vb_paras.E_beta;
      // std::cout << "Previous beta: " << E_beta_prev << std::endl;
      update_Cov_beta();
      // std::cout << "Covariance of beta: " << vb_paras.Cov_beta << std::endl;
      update_E_beta();
      if(vb_control.ELBO_stop == 0){
        update_ELBO();
        // std::cout << "ELBO: " << vb_profile.ELBO << std::endl;
        if(compute_paras_diff(vb_paras.E_beta,E_beta_prev) < vb_control.para_diff_tol){
          save_vb_profile();
          monitor_vb();
          break;
        }
      } else {
        double ELBO_prev = vb_paras.ELBO;
        update_ELBO();
        // sstd::cout << "ELBO: " << vb_profile.ELBO << std::endl;
        if(abs(vb_paras.ELBO - ELBO_prev) < vb_control.ELBO_diff_tol){
          save_vb_profile();
          monitor_vb();
          break;
        }
      }
      // std::cout << "Expectation of beta: " << vb_paras.E_beta << std::endl;
      update_E_inv_a_beta();
      update_E_Xbeta_sq();
      update_E_SSE();
      // std::cout << "SSE: " << vb_paras.E_SSE << std::endl;
      update_E_beta_innerproduct();
      update_E_inv_sigma_sq();
      update_E_inv_sigma_beta_sq();
      

      save_vb_profile();
      monitor_vb();
    }
  };
  
  
  //output for R
  List get_gibbs_post_mean(){
    vec beta = mean(paras_sample.beta,1);
    return List::create(Named("beta") = beta,
                        Named("sigma_sq") = mean(paras_sample.sigma_sq),
                        Named("sigma_beta_sq") = mean(paras_sample.sigma_beta_sq),
                        Named("a_beta") = mean(paras_sample.a_beta));
  };
  
  List get_gibbs_sample(){
    return List::create(Named("beta") = paras_sample.beta,
                        Named("sigma_sq") = paras_sample.sigma_sq,
                        Named("sigma_beta_sq") = paras_sample.sigma_beta_sq,
                        Named("a_beta") = paras_sample.a_beta);
  };
  
  List get_gibbs_trace(){
    uvec iters = linspace<uvec>(1,gibbs_control.total_iter,gibbs_control.total_profile);
    return List::create(Named("iters") = iters,
                        Named("loglik") = gibbs_profile.loglik,
                        Named("logpost") = gibbs_profile.logpost);
  }
  
  List get_gibbs_control(){
    return List::create(Named("total_iter") = gibbs_control.total_iter,
                        Named("burnin") = gibbs_control.burnin,
                        Named("mcmc_sample") = gibbs_control.mcmc_sample,
                        Named("thinning") = gibbs_control.thinning,
                        Named("verbose") = gibbs_control.verbose,
                        Named("save_profile") = gibbs_control.save_profile,
                        Named("total_profile") = gibbs_control.total_profile);
  }
  
  List get_vb_post_mean(){
    return List::create(Named("beta") = vb_paras.E_beta,
                        Named("sigma_sq") = vb_paras.E_SSE / dat.num_subjects,
                        Named("sigma_beta_sq") = (vb_paras.E_beta_innerproduct + 2.0*vb_paras.E_inv_a_beta) / (dat.num_predictors+1),
                        Named("inv_a_beta") = vb_paras.E_inv_a_beta);
  };
  
  List get_vb_trace(){
    int actual_profile_iter = 1;
    if(iter == 0){
      iter = 1;
    }
    if(vb_control.save_profile>0){
      actual_profile_iter = iter/vb_control.save_profile;
    }
    uvec iters = linspace<uvec>(1,iter,actual_profile_iter);
    return List::create(Named("iters") = iters,
                        Named("ELBO") = vb_profile.ELBO.rows(0,actual_profile_iter-1));
  }
  
  List get_vb_control(){
    return List::create(Named("max_iter")= vb_control.max_iter,
    Named("para_diff_tol") = vb_control.para_diff_tol,
    Named("ELBO_stop") = vb_control.ELBO_stop,
    Named("ELBO_diff_tol") = vb_control.ELBO_diff_tol,
    Named("verbose") = vb_control.verbose,
    Named("save_profile") = vb_control.save_profile,
    Named("total_profile") = vb_control.total_profile);
  };
  
  int get_iter(){
    return iter;
  };
};


//[[Rcpp::export]]
List simul_dat_linear(int n, double intercept, vec& beta, double X_rho, double X_sd,
                      double sigma_sq = 1){
  double sqrt_X_rho = sqrt(X_rho);
  mat X = X_sd*sqrt(1.0-X_rho)*randn<mat>(n,beta.n_elem);
  vec Z = X_sd*sqrt_X_rho*randn<vec>(n);
  for(int j=0; j<X.n_cols;j++){
    X.col(j) += Z;
  }
  
  vec mu = intercept + X*beta;
  vec epsilon = sqrt(sigma_sq)*randn<vec>(n);
  vec y = mu + epsilon;
  return List::create(Named("y") = y,
                      Named("X") = X,
                      Named("mu") = mu,
                      Named("intercept") = intercept,
                      Named("beta") = beta,
                      Named("X_rho") = X_rho,
                      Named("X_sd") = X_sd,
                      Named("sigma_sq") = sigma_sq);
}


// [[Rcpp::export]]
List Bayes_linear_reg(vec& y, mat& X, 
                     CharacterVector method,
                     double initial_sigma_sq = 1,
                     double initial_sigma_beta_sq = 1,
                     bool include_intercept = true,
                     int mcmc_sample = 500, 
                     int burnin = 5000, 
                     int thinning = 10,
                     int max_iter = 1000,
                     double paras_diff_tol = 1e-6,
                     int ELBO_stop = 1,
                     double ELBO_diff_tol = 1e-6,
                     int verbose = 5000,
                     int save_profile = 1){
  
  wall_clock timer;
  timer.tic();
  BayesLinearReg model;
  
  if(include_intercept){
    X.insert_cols(0,ones<vec>(X.n_rows));
  }
  
  model.load_data(y,X);
  model.set_method(method);
  
  
  if(model.get_method()==0){
    model.set_gibbs_control(mcmc_sample,
                        burnin,
                        thinning,
                        verbose,
                        save_profile);
  } else if(model.get_method()==1){
    model.set_vb_control(max_iter,
                         paras_diff_tol,
                         ELBO_stop,
                         ELBO_diff_tol,
                            verbose,
                            save_profile);
  }
  
  //std::cout << "set control done" << std::endl;
  
  model.set_paras_initial_values(initial_sigma_sq, initial_sigma_beta_sq);
  
  //std::cout << "set initial values" << std::endl;
  
  if(model.get_method()==0){
    model.run_gibbs(); 
  } else if(model.get_method()==1){
    model.run_mfvb(); 
  }
  
  //std::cout << "mfvb" << std::endl;
  
  double elapsed = timer.toc();
  
  List output;
  
  if(model.get_method()==0){
    output = List::create(Named("post_mean") = model.get_gibbs_post_mean(),
                 Named("mcmc") = model.get_gibbs_sample(),
                 Named("trace") = model.get_gibbs_trace(),
                 Named("mcmc_control") = model.get_gibbs_control(),
                 Named("method") = method,
                 Named("elapsed") = elapsed);
  } else if(model.get_method()==1){
    output = List::create(Named("post_mean") = model.get_vb_post_mean(),
                          Named("iter") = model.get_iter(),
                          Named("trace") = model.get_vb_trace(),
                          Named("vb_control") = model.get_vb_control(),
                          Named("method") = method,
                          Named("elapsed") = elapsed);
  }
  
  
  return output;
  
}