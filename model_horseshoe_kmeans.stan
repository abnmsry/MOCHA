functions {
  real partial_likelihood(array[] int indices_slice, int start, int end,
                          int N, int K, vector y, matrix linear, vector mu_k, vector sqrt_sigma_k) {
    real target_partial = 0.0;
    int current_slice_size = end - start + 1;
    matrix[current_slice_size, K] log_lik_components;
    
    for (i in 1:current_slice_size) {
      int idx = indices_slice[i];
      for (k in 1:K) {
        log_lik_components[i, k] = normal_lpdf(y[idx] | mu_k[k], sqrt_sigma_k[k]);
      }
      real log_normalizer = log_sum_exp(linear[idx]);
      vector[K] log_contrib = (linear[idx] + log_lik_components[i])';
      target_partial += log_sum_exp(log_contrib) - log_normalizer;
    }
    
    return target_partial;
  }
}

data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> K;
  matrix[N,D] X;
  vector[N] y;
  vector[K] mu_prior;
  vector[K] sigma_prior;
  int doparallel;
  int grainsize;
}

transformed data {
  real a = 5;
  real inv_gamma_shape = a;
  vector[K] inv_gamma_scale = (a - 1) * square(sigma_prior);
  array[N] int observation_indices;
  for (i in 1:N) {
    observation_indices[i] = i;
  }
}

parameters {
  // real alpha;
  real<lower=0.001> tau;
  // real alpha_mu;
  // real<lower=0.001> alpha_sigma;
  // vector[K] alpha_offset;
  vector[K] mu_offset;
  vector<lower=0.001>[K] sigma_k;
  vector<lower=0.001>[D] lambda_d;
  matrix[D,K] eta_kd_v;
}

transformed parameters {
  matrix[D, K] eta_kd;
  // vector[K] alpha_k = alpha_mu + alpha_offset * alpha_sigma;
  vector[K] mu_k = mu_prior + mu_offset;
  matrix[N, K] linear;

  eta_kd = diag_pre_multiply(lambda_d, eta_kd_v) * tau;
  // linear = rep_matrix(alpha_k', N) + X * eta_kd;
  // linear = rep_matrix(alpha, N, K) + X * eta_kd;
  linear = X * eta_kd;
}


model {
  // Priors
  tau ~ normal(0, 1);
  // alpha ~ normal(0, 1);
  // alpha_mu ~ normal(0, 1);
  // alpha_sigma ~ inv_gamma(1, 1);
  // alpha_offset ~ std_normal();
  sigma_k ~ inv_gamma(inv_gamma_shape, inv_gamma_scale);
  mu_offset ~ std_normal();
  lambda_d ~ student_t(3, 0, 1);
  // lambda_d ~ cauchy(0, 1);
  to_vector(eta_kd_v) ~ std_normal();

  vector[K] sqrt_sigma_k = sqrt(sigma_k);

  // Likelihood
  if (doparallel) {
    target += reduce_sum(partial_likelihood, observation_indices, grainsize, N, K, y, linear, mu_k, sqrt_sigma_k);
  }
  else {
    target += partial_likelihood(observation_indices, 1, N, N, K, y, linear, mu_k, sqrt_sigma_k);
  }


}


generated quantities {
  matrix[N, K] prob;
  
  for (i in 1:N) {
    prob[i] = softmax(linear[i]')';
  }
  
}
