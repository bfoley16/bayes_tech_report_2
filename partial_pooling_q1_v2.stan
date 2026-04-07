//
// Tech Report 2: Question 1
//

data {
  int<lower=1> N; // total number of observations
  int<lower=1> k; // number of matches
  int<lower=0, upper=1> PowerPlayPresent[N];
  int<lower=0, upper=1> y[N]; 
  int<lower=0> i[N]; // these are our indices for each match
  
  // hyperparameters for regression coefficients
  real u0;
  real u1;
  real<lower=0> v0;
  real<lower=0> v1;
  
  // hyperparameters for random effects
  real<lower=0> alpha0;
  real<lower=0> alpha1;
  real<lower=0> eta0;
  real<lower=0> eta1;
}

parameters {
  // prior layer
  vector[k] beta0;
  vector[k] beta1;
  
  // hyperprior layer
  real m0;
  real m1;
  real<lower=0> s0;
  real<lower=0> s1;
}

transformed parameters {
  // mean model (log(odds))
  vector[N] log_odds;
  
  // probability
  vector[N] prob;
  
  // fill in log_odds for each match
  for (l in 1:N) {
    log_odds[l] = beta0[i[l]] + beta1[i[l]] * PowerPlayPresent[l]; // betas depend on the group. i[l] tells you what group you're in
  }
  
  // convert log odds to probability
  prob = inv_logit(log_odds);
  
  // getting 1/s0^2 and 1/s1^2 for the Gamma hyperpriors
  real<lower=0> s02_inv;
  real<lower=0> s12_inv;
  
  s02_inv = 1/(s0^2);
  s12_inv = 1/(s1^2);
}

model {
  target += bernoulli_lpmf(y | prob);
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | m1, s1);
  target += normal_lpdf(m0 | u0, v0);
  target += normal_lpdf(m1 | u1, v1);
  target += gamma_lpdf(s02_inv | alpha0, eta0);
  target += gamma_lpdf(s12_inv | alpha1, eta1);
}

generated quantities {
  vector[k] exp_beta0;
  vector[k] exp_beta1;
  exp_beta0 = exp(beta0);
  exp_beta1 = exp(beta1);
}

