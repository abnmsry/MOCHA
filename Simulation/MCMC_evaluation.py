import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp,gammaln
import arviz as az

class evaluation:
    def __init__(self, X, y, K, result):
        self.X = X
        self.y = y
        self.K = K
        self.result = result
        self.N = X.shape[0]
        self.D = X.shape[1]


    def index(self):
        loo, dic, waic = self.evaluate_from_netcdf()
        return loo, dic, waic
    

    def evaluate_from_netcdf(self, num_chains=4):

        prob_samples_chains = self.result.posterior["prob"].values  # (num_chains, num_samples_per_chain, N, K)
        mu_k_samples_chains = self.result.posterior["mu_k"].values      # (num_chains, num_samples_per_chain, K)
        sigma_k_samples_chains = self.result.posterior["sigma_k"].values
        
        num_chains = prob_samples_chains.shape[0]
        num_samples_per_chain = prob_samples_chains.shape[1]

        dic_values = []

        # prob_samples = prob_samples_chains.reshape(-1, self.N, self.K)
        # mu_k_samples = mu_k_samples_chains.reshape(-1, self.K)
        # sigma_k_samples = sigma_k_samples_chains.reshape(-1, self.K)


        ## loo
        all_log_lik = np.zeros((4, 1000, self.N))
        for chain in range(num_chains):
            prob_samples = prob_samples_chains[chain,:,:,:]      
            mu_k_samples = mu_k_samples_chains[chain,:,:]
            sigma_k_samples = sigma_k_samples_chains[chain,:,:]

            # 计算对数似然函数
            log_lik = self.calculate_log_lik(prob_samples, mu_k_samples, sigma_k_samples)
            all_log_lik[chain,:,:] = log_lik

        loo = self.calculate_loo(all_log_lik)
        waic = self.calculate_waic_auto(all_log_lik)


        ## dic_alt
        for chain in range(num_chains):
            prob_samples = prob_samples_chains[chain,:,:,:]      
            mu_k_samples = mu_k_samples_chains[chain,:,:]
            sigma_k_samples = sigma_k_samples_chains[chain,:,:]

            # 计算对数似然函数
            log_lik = self.calculate_log_lik(prob_samples, mu_k_samples, sigma_k_samples)
            # 计算 WAIC
            # waic = self.calculate_bic(log_lik)
            dic = self.calculate_dic(log_lik.T,prob_samples, mu_k_samples, sigma_k_samples)
            dic_values.append(dic)
        dic = np.mean(dic_values)

        return loo, dic, waic
    
    def calculate_log_lik(self, prob_samples, mu_k_samples, sigma_k_samples):
        eps = 1e-10
        S, N, _ = prob_samples.shape
        # prob_samples = np.clip(prob_samples, eps, 1 - eps)
        log_contrib = np.log(prob_samples+eps) + norm.logpdf(self.y[np.newaxis, :, np.newaxis],
                                                 loc=mu_k_samples[:, np.newaxis, :],
                                                 scale=np.sqrt(sigma_k_samples[:, np.newaxis, :]))
        log_lik = logsumexp(log_contrib, axis=2)
        return log_lik
    


    def calculate_loo(self, log_lik):
        """Calculates LOO using PSIS (Pareto Smoothed Importance Sampling)."""
        posterior = self.result.posterior
        posterior_samples = {var_name: posterior[var_name].values for var_name in posterior.data_vars}

        inference_data = az.from_dict(
            posterior=posterior_samples,  # 你的后验样本
            log_likelihood={"y": log_lik}   # 关键: 将 log_lik 存储为 log_likelihood 组的一部分
        )

        loo_result = az.loo(inference_data)  # Use arviz library for LOO calculation
        return loo_result.elpd_loo
    

    def calculate_dic(self, log_lik, prob_samples, mu_k_samples, sigma_k_samples):
        # 计算对数似然的后验期望 E[log(p(y | θ))]
        mean_loglik = np.mean(np.sum(log_lik, axis=1))
        # 计算参数后验期望 E[θ]
        mean_prob_k = np.mean(prob_samples, axis=0)
        mean_mu_k = np.mean(mu_k_samples, axis=0)
        mean_sigma_k = np.mean(sigma_k_samples, axis=0)
        # 计算 log(p(y | E[θ]))
        loglik_at_means = self.calculate_log_lik_at_means(mean_prob_k, mean_mu_k, mean_sigma_k)
        
        # 计算有效参数数量 pD
        # pD = 2 * (loglik_at_means - mean_loglik)
        # print(log_lik.shape)
        pD = 2 * np.var(np.sum(log_lik, axis=0))
        # 计算 DIC
        dic = -2 * (loglik_at_means - pD)
        return dic

    def calculate_log_lik_at_means(self, mean_prob_k, mean_mu_k, mean_sigma_k):
        """
        计算在参数后验期望下的对数似然 log(p(y | E[θ])).
        """
        eps = 1e-10
        y_expanded = self.y[:, np.newaxis]  # 形状 (N, 1)
        mu_k_expanded = mean_mu_k[np.newaxis, :]  # 形状 (1, K)
        sigma_k_expanded = mean_sigma_k[np.newaxis, :]  # 形状 (1, K)
        prob_k_expanded = mean_prob_k[np.newaxis, :] # 形状 (1,K)
        # 计算 log_contrib
        # prob_k_expanded = np.clip(prob_k_expanded, eps, 1 - eps)
        log_contrib = np.log(prob_k_expanded + eps) + norm.logpdf(y_expanded, loc=mu_k_expanded, scale=np.sqrt(sigma_k_expanded))
        # 计算 loglik_at_means
        loglik_at_means = np.sum(logsumexp(log_contrib, axis=1))
        return loglik_at_means
   

    def calculate_waic_auto(self, log_lik):
        posterior = self.result.posterior
        posterior_samples = {var_name: posterior[var_name].values for var_name in posterior.data_vars}

        inference_data = az.from_dict(
            posterior=posterior_samples,  # 你的后验样本
            log_likelihood={"y": log_lik}   # 关键: 将 log_lik 存储为 log_likelihood 组的一部分
        )
        
        waic_result = az.waic(inference_data)
        return waic_result.elpd_waic 


    def calculate_waic(self, log_lik):
        lppd = np.sum(np.log(np.mean(np.exp(log_lik), axis=1)))
        # lppd = np.sum(logsumexp(log_lik, axis=1) - np.log(log_lik.shape[0]))
        p_waic_1 = np.sum(np.var(log_lik, axis=1))
        p_waic_2 = 2 * np.sum(np.log(np.mean(np.exp(log_lik),axis=1))-np.mean(log_lik,axis=1))
        waic = -2 * (lppd - p_waic_2)
        return waic

    
