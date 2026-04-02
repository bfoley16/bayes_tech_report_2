data {
  int<lower=1> N; // total number of observations - sets
  int<lower=1> k; // number of matches
  vector[N] x; // this is powerplay
  vector[N] x2; // this is success
  int<lower=0, upper=1> y[N]; 
  int<lower=0> i[N]; // these are our indices for each Match
                      // We need right here our two indicator variables
  
  // hyperparameters for regression coefficients (Betas) and likelihood variance
  real u0;
  real u1;
  real u2;
  real<lower=0> v0;
  real<lower=0> v1;
  real<lower=0> v2;

  
  // hyperparameters for random effects (Gamma funcs)
  real<lower=0> alpha0;
  real<lower=0> alpha1;
  real<lower=0> alpha2;

  real<lower=0> eta0;
  real<lower=0> eta1;
  real<lower=0> eta2;
}

parameters {
  // prior layer
  vector[k] beta0;
  vector[k] beta1;
  vector[k] beta2;
  
  // hyperperior layer
  real m0;
  real m1;
  real m2;
  real<lower=0> s0;
  real<lower=0> s1;
  real<lower=0> s2;
}

transformed parameters {
  // mean model
  vector[N] pie; // pi is yellow cant trust that it wont be read as a command
  //array[N] real pie;
  // fill in pi for each Match
  for (l in 1:N) {
    pie[l] = beta0[i[l]] + beta1[i[l]] * x[l] + beta2[i[l]] * x2[l];
  }
  
  pie = inv_logit(pie);

  
  // getting 1/s0^2 and 1/s1^2 and 1/s2^2 for the Gamma hyperpriors
  real<lower=0> s02_inv;
  real<lower=0> s12_inv;
  real<lower=0> s22_inv;
  
  s02_inv = 1/(s0^2);
  s12_inv = 1/(s1^2);
  s22_inv = 1/(s2^2);
}

model {
  target += bernoulli_lpmf(y | pie);
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | m1, s1);
  target += normal_lpdf(beta2 | m2, s2);
  target += normal_lpdf(m0 | u0, v0);
  target += normal_lpdf(m1 | u1, v1);
  target += normal_lpdf(m2 | u2, v2);
  target += gamma_lpdf(s02_inv | alpha0, eta0);
  target += gamma_lpdf(s12_inv | alpha1, eta1);
  target += gamma_lpdf(s22_inv | alpha2, eta2);
}
