# import matplotlib.pyplot as plt
# # %config InlineBackend.figure_format = 'retina'
# import scienceplots
# plt.style.use(['science', 'nature'])

import numpy as np
import pandas as pd
import arviz as az
# import EM_grouplasso_multimodal
import importlib
import nest_asyncio
nest_asyncio.apply()
import MCMC_evaluation

from scipy.stats import gaussian_kde
from joblib import Parallel, delayed 
import os

from sklearn.metrics import auc

num_cores = os.cpu_count()

def calculate_marginal_map(samples):

    kde = gaussian_kde(samples)
    sample_min = np.min(samples)
    sample_max = np.max(samples)

    range_buffer = (sample_max - sample_min) * 0.1

    if range_buffer == 0:
        range_buffer = 1e-6
    
    eval_range = np.linspace(sample_min - range_buffer, sample_max + range_buffer, 10000)
    
    if len(eval_range) == 0 or np.all(np.isnan(eval_range)):
        return np.nan
    density_values = kde(eval_range)
    
    if len(density_values) == 0 or np.all(np.isnan(density_values)):
        return np.nan
        
    map_estimate = eval_range[np.argmax(density_values)]
    return map_estimate

def calculate_auc(select_scores, true_labels, num_thresholds=1000):

    min_lamb = min(select_scores)
    max_lamb = max(select_scores)

    thresholds = np.linspace(min_lamb, max_lamb, 1000)
    tpr_list = []
    fpr_list = []

    for t in thresholds:
        y_pred = (select_scores > t).astype(int)
        TP = np.sum((y_pred == 1) & (true_labels == 1))
        FP = np.sum((y_pred == 1) & (true_labels == 0))
        FN = np.sum((y_pred == 0) & (true_labels == 1))
        TN = np.sum((y_pred == 0) & (true_labels == 0))

        TPR = TP / (TP + FN + 1e-8)
        FPR = FP / (FP + TN + 1e-8)

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # AUC
    manual_auc = auc(fpr_list, tpr_list)

    youden_j = np.array(tpr_list) - np.array(fpr_list)
    opt_index = np.argmax(youden_j)
    opt = thresholds[opt_index]
    # opt_tpr = tpr_list[opt_index]
    # opt_fpr = fpr_list[opt_index]

    # print(f"\nOptimal threshold (Youden's J): {opt:.4f}")
    # print(f"Corresponding TPR: {opt_tpr:.4f}")
    # print(f"Corresponding FPR: {opt_fpr:.4f}")

    return manual_auc, opt 


# store results
results = []
coverage = []
mu_sigma_results_K3 = []
mu_sigma_results_K5 = []
D = 500
for K in [3,5]:
    for n in [200,500]:
        for r in [1,5,10]:
            for tr in [1,5]:
                for i in range(100):

                    print(f"Start: k{K}n{n}r{r}tr{tr} i={i}", flush=True)
                    
                    # 0. data loading
                    result_dir = f"./Simulation_results/Simulation_2/MCMC_results/k{K}n{n}r{r}tr{tr}/"
                    result = az.from_netcdf(result_dir + f"dat_{i}.nc")

                    dat_dir = f"./Simulation_2/k{K}n{n}r{r}tr{tr}/"
                    sim_dat = np.load(dat_dir + f"dat_{i}.npz")

                    # 1. compare distributional estimation
                    # 1.1 mu
                    mu_true = sim_dat["mu"]
                    indice_true = np.argsort(mu_true)
                    mu_true = mu_true[indice_true]
                    sigma_true = sim_dat["sigma"][indice_true]

                    mu_samples = result.posterior['mu_k'].values.reshape(4000, K)
                    mu_est = Parallel(n_jobs=num_cores)(
                                delayed(calculate_marginal_map)(mu_samples[:, i]) for i in range(K)
                            )
                    mu_est = np.array(mu_est)

                    indice_est = np.argsort(mu_est)
                    mu_est = mu_est[indice_est]

                    # 1.2 sigma
                    sigma_samples = result.posterior['sigma_k'].values.reshape(4000, K)
                    sigma_est = Parallel(n_jobs=num_cores)(
                                delayed(calculate_marginal_map)(sigma_samples[:, i]) for i in range(K)
                            )
                    sigma_est = np.array(sigma_est)
                    sigma_est = sigma_est[indice_est]

                    # 1.3 RRMSE
                    mu_sigma_data = {
                        'K': K, 'n': n, 'r': r, 'tr': tr, 'i': i
                    }
                    for k_idx in range(K):
                        mu_sigma_data[f'mu_est_{k_idx+1}'] = mu_est[k_idx]
                        mu_sigma_data[f'sigma_est_{k_idx+1}'] = sigma_est[k_idx]
                        mu_sigma_data[f'mu_true_{k_idx+1}'] = mu_true[k_idx]
                        mu_sigma_data[f'sigma_true_{k_idx+1}'] = sigma_true[k_idx]
                    
                    if K == 3:
                        mu_sigma_results_K3.append(mu_sigma_data)
                    elif K == 5:
                        mu_sigma_results_K5.append(mu_sigma_data)

                    # rmse_mu = np.sqrt(np.mean((mu_true - mu_est)**2))
                    # rmse_sigma = np.sqrt(np.mean((sigma_true-sigma_est)**2))

                    # rrmse_mu = np.sqrt(np.mean((mu_true - mu_est)**2))/np.mean(mu_true)
                    # rrmse_sigma = np.sqrt(np.mean((sigma_true-sigma_est)**2))/np.mean(sigma_true)

                    # print('RMSE:', rmse_mu, rmse_sigma, flush=True)
                    # print('RRMSE:', rrmse_mu, rrmse_sigma, flush=True)
                    # print(mu_true, sigma_true)
                    # print(mu_est, sigma_est)
                    # print(sim_dat["theta"],sum(sim_dat["w"]))


                    # 2.1 get AUC
                    lambda_samples = result.posterior['lambda_d'].values.reshape(4000, D+1)
                    select = Parallel(n_jobs=num_cores)(
                                delayed(calculate_marginal_map)(lambda_samples[:, i]) for i in range(D)
                            )
                    select = np.array(select)
                    truth = sim_dat["w"]

                    total_auc, total_youden = calculate_auc(select,truth)
                    D_genotype = round(D * r / (r + 1))
                    genotype_auc, _ = calculate_auc(select[np.arange(D_genotype)], truth[np.arange(D_genotype)])
                    mrna_auc, _ = calculate_auc(select[np.arange(D_genotype,D)], truth[np.arange(D_genotype,D)])
                    print("AUC: ",total_auc, "genotype_AUC: ",genotype_auc,"mrna_AUC: ",mrna_auc,flush=True)
                    print(f"Optimal threshold: {total_youden:.4f}")

                    # 2.2 power and Type-I-error compared to EM
                    threshold = 0.5218048
                    y_pred_tmp = (select > threshold).astype(int)
                    TP = np.sum((y_pred_tmp == 1) & (truth == 1))
                    FP = np.sum((y_pred_tmp == 1) & (truth == 0))
                    FN = np.sum((y_pred_tmp == 0) & (truth == 1))
                    TN = np.sum((y_pred_tmp == 0) & (truth == 0))

                    Type_I_error = FP / (FP + TN) 
                    Power = TP / (TP + FN)
                    print("Type_I_error: ",Type_I_error,"Power: ",Power)

                    tmp_result = [K,n,r,tr,i,total_auc,genotype_auc,mrna_auc,total_youden,Type_I_error,Power,TP,FP,FN,TN]
                    results.append(tmp_result)

                    # plt.figure()
                    # plt.plot(fpr_list, tpr_list, color='darkorange', lw=2)
                    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 绘制对角线

                    # plt.text(0.95, 0.05, f'AUC={manual_auc:.2f}',
                    #         horizontalalignment='right', # 水平右对齐
                    #         verticalalignment='bottom',  # 垂直底部对齐
                    #         fontsize=10,
                    #         color='black') # 添加白色背景框，增加可读性

                    # # plt.scatter(opt_fpr, opt_tpr, color='red', s=100, marker='o', label=f'Optimal (Youden\'s J): {opt:.2f}')

                    # plt.xlim([0.0, 1.0])
                    # plt.ylim([0.0, 1.05])
                    # plt.xlabel('False Positive Rate',fontsize=10)
                    # plt.ylabel('True Positive Rate',fontsize=10)
                    # plt.xticks(fontsize=10)
                    # plt.yticks(fontsize=10) 
                    # # plt.legend(loc="lower right",fontsize=18)
                    # plt.grid(True)
                    # # plt.savefig(f'Fig_MCMC/AUC_k{K}n{N}r{r}tr{tr}.pdf', format='pdf', bbox_inches='tight', transparent=True)

                    # # plt.show()


                    # 3. eta_kd coverage
                    # eta_kd_samples = result.posterior['eta_kd'].values
                    # eta_kd_samples = eta_kd_samples[:, :, :-1, :]

                    # eta_kd_samples = eta_kd_samples.reshape(4000, D, K)

                    # eta_kd_map = Parallel(n_jobs=num_cores)(
                    #     delayed(calculate_marginal_map)(eta_kd_samples[:, i, j]) 
                    #     for i in range(D) 
                    #     for j in range(K)
                    # )

                    # eta_kd_map = np.array(eta_kd_map).reshape((D, K))


                    # order indices
                    eta_kd_true = sim_dat["eta"]
                    eta_kd_true = eta_kd_true[indice_true,:]


                    # hdi_prob = 95%
                    hdi_95 = az.summary(result, var_names=["eta_kd"], hdi_prob=0.95)
                    hdi_95_lr = hdi_95["hdi_2.5%"].values
                    hdi_95_lr = hdi_95_lr.reshape((D+1, K))
                    hdi_95_lr = hdi_95_lr[:-1,:]
                    hdi_95_lr = hdi_95_lr[:,indice_est]
                    hdi_95_lr = hdi_95_lr.T

                    hdi_95_ur = hdi_95["hdi_97.5%"].values
                    hdi_95_ur = hdi_95_ur.reshape((D+1, K))
                    hdi_95_ur = hdi_95_ur[:-1,:]
                    hdi_95_ur = hdi_95_ur[:,indice_est]
                    hdi_95_ur = hdi_95_ur.T

                    coverage_mask = (eta_kd_true >= hdi_95_lr) & (eta_kd_true <= hdi_95_ur)
                    coverage_count = np.sum(coverage_mask)

                    print("Coverage count: ", coverage_count)

                    tmp_result = [coverage_count]
                    coverage.append(tmp_result)

                

column_names = ['k','n','r','tr','i','total_auc','genotyp_auc','mrna_auc','total_youden','Type_I_error','Power','TP','FP','FN','TN']
results_df = pd.DataFrame(results, columns=column_names)
results_df.to_csv('./Simulation_2/MCMC_summary.tsv', sep='\t', index=False)

if mu_sigma_results_K3:
    mu_sigma_df_K3 = pd.DataFrame(mu_sigma_results_K3)
    mu_sigma_df_K3.to_csv('./Simulation_2/MCMC_mu_sigma_K3.tsv', sep='\t', index=False)
    print("Saved MCMC_mu_sigma_K3.tsv")
# Save mu and sigma results for K=5
if mu_sigma_results_K5:
    mu_sigma_df_K5 = pd.DataFrame(mu_sigma_results_K5)
    mu_sigma_df_K5.to_csv('./Simulation_2/MCMC_mu_sigma_K5.tsv', sep='\t', index=False)
    print("Saved MCMC_mu_sigma_K5.tsv")


column_names = ['Coverage_count']
results_df = pd.DataFrame(coverage, columns=column_names)
results_df.to_csv('./Simulation_2/MCMC_coef_coverage.tsv', sep='\t', index=False)
