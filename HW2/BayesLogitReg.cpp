#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


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
double adjust_acceptance(double accept,double sgm,double target = 0.5){
  double y = 1. + 1000.*(accept-target)*(accept-target)*(accept-target);
  if (y < .9)
    y = .9;
  if (y > 1.1)
    y = 1.1;
  sgm *= y;
  return sgm;
}



class BayesProbitReg{
  
private:
  int method;

  struct ProbitRegData{
    int num_subjects;
    int num_predictors;
    vec y;
    mat X;
    vec Xty;
    double sum_y;
  } dat;
  
  struct ProbitVars{
    vec X_beta;
    vec prob;
    double sum_beta_sq;
  } current_vars, proposal_vars;
  
  struct ProbitRegParas{
    vec beta;
    vec momentum;
    double sigma2_beta;
  } current_paras, proposal_paras, initial_paras;
  
  struct logDensities{
    double post;
    double prior;
    double lik;
  } current_logden, proposal_logden;
  
  struct DerivLogDensities{
    vec post;
    vec prior;
    vec lik; 
  } current_d_logden, proposal_d_logden;
  
  struct sample{
    mat beta;
    vec sigma2_beta;
  } mcmc;
  
  struct profile{
    vec loglik;
    vec logprior;
    vec logpost;
    vec accept_rate;
  } trace;
  
  struct algorithmControl{
    int mcmc_sample;
    int burnin;
    int thinning;
    int step_adjust_accept;
    int maxiter_adjust_accept;
    double initial_step_size;
    double step_size;
    double step_size_sq;
    double target_accept;
    int leapfrog_steps;
    int verbose;
  } control;
  
  
  double log_accept_ratio;
  double accept_rate;
  int accept_count;
  int iter;
  int total_iter;

  
  
public:
  
  void set_method(CharacterVector);
  void load_data(const vec&, const mat&);
  void set_initial_values(const vec&, const double&);
  void set_alg_control(const int&, const int&, const int&, const int&, const int&,
                       const double&,const double&, const int&, const int&);
  double comp_loglik(const vec&);
  double get_proposal_loglik(){
    return proposal_logden.lik;
  };

  void update_proposal_paras();
  void update_proposal_vars();
  void update_proposal_loglik();
  void update_proposal_logprior();
  void update_proposal_logden();
  void update_proposal_d_loglik();
  void update_proposal_d_logprior();
  void update_proposal_d_logden();
  
  
  
  void update_log_accept_ratio();
  void update_current_paras();
  
  void update_accept_rate();
  void update_step_size();
  
  void save_profile();
  void save_mcmc_sample();
  
  void monitor_mcmc();

  void run_mcmc();
  
  List get_control();
  List get_mcmc();
  List get_post_mean();
  List get_trace();

};

void BayesProbitReg::set_method(CharacterVector in_method){
  
  if(in_method(0)=="RW"){
    std::cout << "Random Walk" << std::endl;
    method = 0;
  } else if(in_method(0)=="MALA"){
    std::cout << "Metropolis-adjusted Langevin Algorithm" << std::endl;
    method = 1;
  } else if(in_method(0)=="HMC"){
    std::cout << "Hamiltonian Monte Carlo" << std::endl;
    method = 2;
  }
}

void BayesProbitReg::load_data(const vec& y, const mat& X){
  dat.num_subjects = y.n_elem;
  dat.num_predictors = X.n_cols;
  dat.y = y;
  dat.X = X;
  dat.Xty = X.t()*y;
  dat.sum_y = accu(y);
}

void BayesProbitReg::set_initial_values(const vec& beta, const double& sigma2_beta){
  
  initial_paras.beta = beta; 
  initial_paras.sigma2_beta = sigma2_beta;
  
  proposal_d_logden.lik.zeros(beta.n_elem);
  proposal_d_logden.prior.zeros(beta.n_elem);
  proposal_d_logden.post.zeros(beta.n_elem);
  
  
  proposal_paras = initial_paras;
  update_proposal_vars();
  update_proposal_logden();
  if(method>0){
    update_proposal_d_logden();
  }
  
  accept_count = 0;
  accept_rate = 0.0;
  log_accept_ratio = 0.0;
  
  current_paras = proposal_paras;
  current_logden = proposal_logden;
  current_d_logden = proposal_d_logden;
  
}

void BayesProbitReg::set_alg_control(const int& mcmc_sample, 
                                    const int& burnin, 
                                    const int& thinning,
                                    const int& step_adjust_accept,
                                    const int& maxiter_adjust_accept,
                                    const double& target_accept,
                                    const double& initial_step_size,
                                    const int& leapfrog_steps,
                                    const int& verbose){
  control.mcmc_sample = mcmc_sample;
  control.burnin = burnin;
  control.thinning = thinning;
  control.maxiter_adjust_accept = maxiter_adjust_accept;
  control.step_adjust_accept = step_adjust_accept;
  control.target_accept = target_accept;
  control.initial_step_size = initial_step_size;
  control.step_size = initial_step_size;
  control.step_size_sq = control.step_size*control.step_size;
  control.leapfrog_steps = leapfrog_steps;
  control.verbose = verbose;
  
  
  //initialize mcmc object
  total_iter = control.burnin + control.mcmc_sample*control.thinning;
  mcmc.beta.zeros(dat.num_predictors,control.mcmc_sample);
  mcmc.sigma2_beta.zeros(control.mcmc_sample);
  
  trace.accept_rate.zeros(total_iter);
  trace.loglik.zeros(total_iter);
  trace.logprior.zeros(total_iter);
  trace.logpost.zeros(total_iter);
  

}

double BayesProbitReg::comp_loglik(const vec& beta){
  vec eta = dat.X*beta;
  eta = log1pexp_fast(eta);
  double loglik = accu(beta%dat.Xty);
  loglik -= accu(eta);
  return(loglik);
}




void BayesProbitReg::update_proposal_paras(){
  //Random Walk
  if(method==0){
    proposal_paras.beta = current_paras.beta + control.step_size*randn<vec>(dat.num_predictors);
    update_proposal_vars();
  }
  
  //MALA
  if(method==1){
    proposal_paras.beta = current_paras.beta;
    proposal_paras.beta += 0.5*control.step_size_sq*current_d_logden.post;
    proposal_paras.beta += control.step_size*randn<vec>(dat.num_predictors);
    update_proposal_vars();
    update_proposal_d_logden();
  }
  
  //HMC
  if(method==2){
    proposal_paras.momentum = randn<vec>(dat.num_predictors);
    current_paras.momentum = proposal_paras.momentum;
    proposal_paras.beta = current_paras.beta;
    update_proposal_vars();
    update_proposal_d_logden();
    proposal_paras.momentum += 0.5*control.step_size*proposal_d_logden.post;
    for(int l=0; l<control.leapfrog_steps;l++){
      proposal_paras.beta +=  control.step_size*proposal_paras.momentum; 
      if(l < control.leapfrog_steps - 1){
        update_proposal_vars();
        update_proposal_d_logden();
        proposal_paras.momentum += 0.5*control.step_size*proposal_d_logden.post;
      }
    }
    update_proposal_vars();
    update_proposal_d_logden();
  }
  
  update_proposal_logden();
}


void BayesProbitReg::update_proposal_vars(){
  proposal_vars.X_beta = dat.X*proposal_paras.beta;
  proposal_vars.sum_beta_sq = sum(proposal_paras.beta%proposal_paras.beta);
  if(method>0){
    proposal_vars.prob = 1.0/(1.0 + exp(-proposal_vars.X_beta));
  }
}


void BayesProbitReg::update_proposal_loglik(){
  vec eta = log1pexp_fast(proposal_vars.X_beta);
  proposal_logden.lik = -accu(eta); 
  if(dat.num_predictors > dat.num_subjects){
    proposal_logden.lik += accu(dat.y%proposal_vars.X_beta);
  } else{
    proposal_logden.lik += accu(proposal_paras.beta%dat.Xty);
  }
}

void BayesProbitReg::update_proposal_d_loglik(){
  for(int j=0;j<dat.num_predictors;j++){
    proposal_d_logden.lik(j) = accu((dat.y - proposal_vars.prob)%dat.X.col(j));
  }
}


void BayesProbitReg::update_proposal_logprior(){
  proposal_logden.prior = -0.5*proposal_vars.sum_beta_sq/proposal_paras.sigma2_beta;
}

void BayesProbitReg::update_proposal_d_logprior(){
  proposal_d_logden.prior = -proposal_paras.beta/proposal_paras.sigma2_beta;
}


void BayesProbitReg::update_proposal_logden(){
  update_proposal_loglik();
  update_proposal_logprior();
  proposal_logden.post = proposal_logden.lik + proposal_logden.prior;
}

void BayesProbitReg::update_proposal_d_logden(){
  update_proposal_d_loglik();
  update_proposal_d_logprior();
  proposal_d_logden.post = proposal_d_logden.lik + proposal_d_logden.prior;
}

void BayesProbitReg::update_log_accept_ratio(){
  log_accept_ratio = proposal_logden.post - current_logden.post;
  if(method==1){
    vec temp = proposal_paras.beta - current_paras.beta - 0.5*control.step_size_sq*current_d_logden.post;
    log_accept_ratio += 0.5*accu(temp%temp)/control.step_size_sq;
    temp = current_paras.beta - proposal_paras.beta - 0.5*control.step_size_sq*proposal_d_logden.post; 
    log_accept_ratio -= 0.5*accu(temp%temp)/control.step_size_sq;
  }
  if(method==2){
    log_accept_ratio += 0.5*accu(current_paras.momentum%current_paras.momentum);
    log_accept_ratio -= 0.5*accu(proposal_paras.momentum%proposal_paras.momentum);
  }
}

void BayesProbitReg::update_current_paras(){
  
  //update beta;
  double u = randu<double>();
  if(log(u) < log_accept_ratio){
    current_paras = proposal_paras;
    current_logden = proposal_logden;
    if(method > 0){
      current_d_logden = proposal_d_logden;
    }
    accept_count++;
  }
  
  //update sigma2_beta;
  proposal_paras = current_paras;
  update_proposal_vars();
  double sum_beta_sq = accu(current_paras.beta%current_paras.beta);
  proposal_paras.sigma2_beta = 1.0/randg(distr_param(0.5*dat.num_predictors + 0.01,
                                                     1.0/(0.5*sum_beta_sq + 0.01)));
  
  update_proposal_logden();
  if(method > 0){
    update_proposal_d_logden();
  }
  
  current_paras.sigma2_beta = proposal_paras.sigma2_beta;
  current_logden = proposal_logden;
  if(method > 0)
    current_d_logden = proposal_d_logden;
  
}

void BayesProbitReg::update_accept_rate(){
  if(iter % control.step_adjust_accept==0){
    accept_rate = accept_count*1.0/control.step_adjust_accept;
    accept_count = 0;
  }
}

void BayesProbitReg::update_step_size(){
  if(iter < control.maxiter_adjust_accept){
    if(iter % control.step_adjust_accept==0){
      control.step_size = adjust_acceptance(accept_rate,
                                            control.step_size,
                                            control.target_accept);
      control.step_size_sq = control.step_size*control.step_size;
    }
  }
}

void BayesProbitReg::save_profile(){
  trace.accept_rate(iter) = accept_rate;
  trace.loglik(iter) = current_logden.lik;
  trace.logprior(iter) = current_logden.prior;
  trace.logpost(iter) = current_logden.post;
}

void BayesProbitReg::save_mcmc_sample(){
  if(iter > control.burnin){
  if((iter - control.burnin)%control.thinning==0){
    int mcmc_iter = (iter - control.burnin)/control.thinning;
    mcmc.beta.col(mcmc_iter) = current_paras.beta;
    mcmc.sigma2_beta(mcmc_iter) = current_paras.sigma2_beta;
  }
  }
}

void BayesProbitReg::monitor_mcmc(){
  if(control.verbose > 0){
    if(iter%control.verbose==0){
      std::cout << "iter: " << iter <<  " logpost: "<<  current_logden.post << std::endl;
    }
  }
  
}
List BayesProbitReg::get_control(){
  return List::create(Named("mcmc_sample") = control.mcmc_sample,
                      Named("burnin") = control.burnin,
                      Named("step_adjust_accept") = control.step_adjust_accept,
                      Named("maxiter_adjust_accept") = control.maxiter_adjust_accept,
                      Named("target_accept") = control.target_accept,
                      Named("initial_step_size") = control.initial_step_size,
                      Named("step_size") = control.step_size,
                      Named("leapfrog_steps") = control.leapfrog_steps,
                      Named("verbose") = control.verbose);
}

void BayesProbitReg::run_mcmc(){
  for(iter=0;iter<total_iter; iter++){
    update_proposal_paras();
    update_log_accept_ratio();
    update_current_paras();
    update_accept_rate();
    update_step_size();
    save_profile();
    save_mcmc_sample();
    monitor_mcmc();
  }
}

List BayesProbitReg::get_mcmc(){
  return List::create(Named("beta") = mcmc.beta,
                      Named("sigma2_beta") = mcmc.sigma2_beta);
}

List BayesProbitReg::get_post_mean(){
  vec beta = mean(mcmc.beta,1);
  return List::create(Named("beta") = beta,
                      Named("sigma2_beta") = mean(mcmc.sigma2_beta));
}

List BayesProbitReg::get_trace(){
  return List::create(Named("accept_rate") = trace.accept_rate,
                      Named("loglik") = trace.loglik,
                      Named("logprior") = trace.logprior,
                      Named("logpost") = trace.logpost);
}


//[[Rcpp::export]]
List simul_dat_Probit(int n, double intercept, vec& beta, double X_rho, double X_sd){
  double sqrt_X_rho = sqrt(X_rho);
  mat X = X_sd*sqrt(1.0-X_rho)*randn<mat>(n,beta.n_elem);
  vec Z = X_sd*sqrt_X_rho*randn<vec>(n);
  for(int j=0; j<X.n_cols;j++){
    X.col(j) += Z;
  }
  vec prob = 1.0/(1.0+exp(-intercept - X*beta));
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
}


// [[Rcpp::export]]
List Bayes_Probit_reg(vec& y, mat& X, 
                     CharacterVector method,
                     vec initial_beta,
                     double initial_sigma2_beta,
                     bool include_intercept = true,
                     double initial_intercept = 0, 
                     int mcmc_sample = 500, 
                     int burnin = 5000, 
                     int thinning = 10,
                     int step_adjust_accept = 100,
                     int maxiter_adjust_accept = 5000,
                     double target_accept = 0.50,
                     double initial_step_size=0.001,
                     int leapfrog_steps = 20,
                     int verbose = 5000){
  
  wall_clock timer;
  timer.tic();
  BayesProbitReg model;
  
  if(include_intercept){
     X.insert_cols(0,ones<vec>(X.n_rows));
     initial_beta.insert_rows(0,initial_intercept*ones<vec>(1));
  }
   
   model.load_data(y,X);
   model.set_method(method);

   model.set_alg_control(mcmc_sample,
                         burnin,
                         thinning,
                         step_adjust_accept,
                         maxiter_adjust_accept,
                         target_accept,
                         initial_step_size,
                         leapfrog_steps,
                         verbose);
   
  model.set_initial_values(initial_beta,
                            initial_sigma2_beta);
  model.run_mcmc(); 
  
  double elapsed = timer.toc();

  
  return List::create(Named("post_mean") = model.get_post_mean(),
                      Named("mcmc") = model.get_mcmc(),
                      Named("trace") = model.get_trace(),
                      Named("mcmc_control") = model.get_control(),
                      Named("method") = method,
                      Named("elapsed") = elapsed);
  
}



