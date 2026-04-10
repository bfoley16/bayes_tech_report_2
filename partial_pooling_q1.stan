//
// Tech Report 2: Question 1
//

data {
  int<lower=1> N; // total number of observations
  int<lower=1> k; // number of matches
  int<lower=0, upper=1> PowerPlay_5[N];
  int<lower=0, upper=1> PowerPlay_6[N];
  int<lower=0, upper=1> PowerPlay_7[N];
  int<lower=0, upper=1> PowerPlay_8[N];
  int<lower=0, upper=9> NumberEnds[N];
  int<lower=0, upper=1> y[N]; 
  int<lower=0> i[N]; // these are our indices for each match
  
  // hyperparameters for regression coefficients
  real u0;
  real<lower=0> v0;
  
  // hyperparameters for random intercept
  real<lower=0> alpha0;
  real<lower=0> eta0;
}

parameters {
  // prior layer
  vector[k] beta0;
  real beta1;
  real beta2;
  real beta3;
  real beta4;
  real beta5;
  
  // hyperprior layer
  real m0;
  real<lower=0> s0;
}

transformed parameters {
  // mean model (log(odds))
  vector[N] log_odds;
  
  // probability
  vector[N] prob;
  
  // fill in log_odds for each match
  for (l in 1:N) {
    log_odds[l] = beta0[i[l]] + beta1 * PowerPlay_5[l] + beta2 * PowerPlay_6[l] + beta3 * PowerPlay_7[l] + beta4 * PowerPlay_8[l] + beta5 * NumberEnds[l]; // betas depend on the group. i[l] tells you what group you're in
  }
  
  // convert log odds to probability
  prob = inv_logit(log_odds);
  
  // getting 1/s0^2 and 1/s1^2 for the Gamma hyperpriors
  real<lower=0> s02_inv;
  
  s02_inv = 1/(s0^2);
}

model {
  target += bernoulli_lpmf(y | prob);
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | -3.037, 0.1);
  target += normal_lpdf(beta2 | -1.35, 0.02);
  target += normal_lpdf(beta3 | -0.906, 0.012);
  target += normal_lpdf(beta4 | 0.69, 0.025);
  target += normal_lpdf(beta5 | -2, 0.5);
  target += normal_lpdf(m0 | u0, v0);
  target += gamma_lpdf(s02_inv | alpha0, eta0);
}

generated quantities {
  vector[k] exp_beta0;
  real exp_beta1;
  real exp_beta2;
  real exp_beta3;
  real exp_beta4;
  real exp_beta5;
  exp_beta0 = exp(beta0);
  exp_beta1 = exp(beta1);
  exp_beta2 = exp(beta2);
  exp_beta3 = exp(beta3);
  exp_beta4 = exp(beta4);
  exp_beta5 = exp(beta5);
}


