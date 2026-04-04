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
  int<lower=0, upper=1> y[N]; 
  int<lower=0> i[N]; // these are our indices for each match
  
  // hyperparameters for regression coefficients
  real u0;
  real u1;
  real u2;
  real u3;
  real u4;
  real<lower=0> v0;
  real<lower=0> v1;
  real<lower=0> v2;
  real<lower=0> v3;
  real<lower=0> v4;
  
  // hyperparameters for random effects
  real<lower=0> alpha0;
  real<lower=0> alpha1;
  real<lower=0> alpha2;
  real<lower=0> alpha3;
  real<lower=0> alpha4;
  real<lower=0> eta0;
  real<lower=0> eta1;
  real<lower=0> eta2;
  real<lower=0> eta3;
  real<lower=0> eta4;
}

parameters {
  // prior layer
  vector[k] beta0;
  vector[k] beta1;
  vector[k] beta2;
  vector[k] beta3;
  vector[k] beta4;
  
  // hyperprior layer
  real m0;
  real m1;
  real m2;
  real m3;
  real m4;
  real<lower=0> s0;
  real<lower=0> s1;
  real<lower=0> s2;
  real<lower=0> s3;
  real<lower=0> s4;
}

transformed parameters {
  // mean model (log(odds))
  vector[N] log_odds;
  
  // probability
  vector[N] prob;
  
  // fill in log_odds for each match
  for (l in 1:N) {
    log_odds[l] = beta0[i[l]] + beta1[i[l]] * PowerPlay_5[l] + beta2[i[l]] * PowerPlay_6[l] + beta3[i[l]] * PowerPlay_7[l] + beta4[i[l]] * PowerPlay_8[l]; // betas depend on the group. i[l] tells you what group you're in
  }
  
  // convert log odds to probability
  prob = inv_logit(log_odds);
  
  // getting 1/s0^2 and 1/s1^2 for the Gamma hyperpriors
  real<lower=0> s02_inv;
  real<lower=0> s12_inv;
  real<lower=0> s22_inv;
  real<lower=0> s32_inv;
  real<lower=0> s42_inv;
  
  s02_inv = 1/(s0^2);
  s12_inv = 1/(s1^2);
  s22_inv = 1/(s2^2);
  s32_inv = 1/(s3^2);
  s42_inv = 1/(s4^2);
}

model {
  target += bernoulli_lpmf(y | prob);
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | m1, s1);
  target += normal_lpdf(beta2 | m2, s2);
  target += normal_lpdf(beta3 | m3, s3);
  target += normal_lpdf(beta4 | m4, s4);
  target += normal_lpdf(m0 | u0, v0);
  target += normal_lpdf(m1 | u1, v1);
  target += normal_lpdf(m2 | u2, v2);
  target += normal_lpdf(m3 | u3, v3);
  target += normal_lpdf(m4 | u4, v4);
  target += gamma_lpdf(s02_inv | alpha0, eta0);
  target += gamma_lpdf(s12_inv | alpha1, eta1);
  target += gamma_lpdf(s22_inv | alpha2, eta2);
  target += gamma_lpdf(s32_inv | alpha3, eta3);
  target += gamma_lpdf(s42_inv | alpha4, eta4);
}

