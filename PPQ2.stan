data {
  int<lower=1> N; // total number of observations - ends
  int<lower=1> k; // number of matches
  vector<lower=0, upper = 1>[N] pp1; // this is PPEnd5
  vector<lower=0, upper = 1>[N] pp2; // this is PPEnd6
  vector<lower=0, upper = 1>[N] pp3; // this is PPEnd7
  vector<lower=0, upper = 1>[N] pp4; // this is PPEnd8
  vector<lower=0, upper = 1>[N] x5; // this is PPsuccess
  vector[N] x6; // NumSuccessfullNonPowerplay
  int<lower=0, upper=1> y[N]; 
  int<lower=0> i[N]; // these are our indices for each Match

  // hyperparameters for regression coefficients (Betas) and likelihood variance
  real u0;

  
  real<lower=0> v0;


}

parameters {
  // prior layer
  vector[k] beta0;
  real beta1;
  real beta2;
  real beta3;
  real beta4;
  real beta5;
  real beta6;

  
  // hyperperior layer
  real m0;

  real<lower=0> s0;

}

transformed parameters {
  // mean model
  vector[N] pie; // pi is yellow cant trust that it wont be read as a command
  // fill in pi for each Match
  for (l in 1:N) {
    pie[l] = beta0[i[l]] + beta1 * pp1[l] + beta2 * pp2[l] + beta3 * pp3[l] + beta4 * pp4[l] + beta5 * x5[l] + beta6 * x6[l];
  }
  
  pie = inv_logit(pie);

  
  // getting 1/s0^2 and 1/s1^2 and 1/s2^2 for the Gamma hyperpriors
  real<lower=0> s02_inv;


  s02_inv = 1/(s0^2);


}

model {
  target += bernoulli_lpmf(y | pie);
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | -3.037, 0.1);
  target += normal_lpdf(beta2 | -1.35, 0.02);
  target += normal_lpdf(beta3 | -0.906, 0.012);
  target += normal_lpdf(beta4 | 0.69, 0.025);
  target += normal_lpdf(beta5 | 0.54, 0.25);
  target += normal_lpdf(beta6 | 0.49, 0.25);

  
  target += normal_lpdf(m0 | 0.6931472, 0.025); #Put priors here
  target += gamma_lpdf(s02_inv | 25, 1000); #Put priors here

}
