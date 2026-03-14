import numpy as np
import pandas as pd
import arviz as az
# import EM_grouplasso_multimodal
import importlib
import nest_asyncio
nest_asyncio.apply()
# import MCMC_evaluation

results = []
mu_sigma_results_K3 = []
mu_sigma_results_K5 = []
D = 500
for K in [3,5]:
    for n in [200,500]:
        for r in [1,5,10]:
            for tr in [1,5]:
                for i in range(100):

                    print(f"Start: k{K}n{n}r{r}tr{tr} i={i}", flush=True)


                    # 2. compare with the true values
                    dat_dir = f"./Simulation/Simulation_2/k{K}n{n}r{r}tr{tr}/"
                    sim_dat = np.load(dat_dir + f"dat_{i}.npz")

                    result_dir = f"./Simulation_results/Simulation_2/EM_results/k{K}n{n}r{r}tr{tr}/"
                    result = np.load(result_dir + f"dat_{i}.npz")

                    mu_true = sim_dat["mu"]
                    indice_true = np.argsort(mu_true)
                    mu_true = mu_true[indice_true]
                    sigma_true = sim_dat["sigma"][indice_true]

                    mu_est = np.sort(result["mu"])
                    indice_est = np.argsort(mu_est)
                    mu_est = mu_est[indice_est]
                    sigma_est = result["sigma"][indice_est]

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
                    # rrmse_mu = np.sqrt(np.mean((mu_true - mu_est)**2))/(np.mean(mu_true))
                    # rrmse_sigma = np.sqrt(np.mean((sigma_true-sigma_est)**2))/np.mean(sigma_true)
                    # print('RMSE:', rmse_mu, rmse_sigma)
                    # print('RRMSE:', rrmse_mu, rrmse_sigma)
                    # print(mu_true, sigma_true)
                    # print(mu_est, sigma_est)

                    truth = sim_dat["w"]
                    truth = set(np.where(truth==1)[0].tolist())
                    select = set(result["select"])
                    select.discard(D)
                    # print(len(truth))
                    # print(len(select))
                    all_indices_set = set(range(D))

                    TP = len(truth.intersection(select))
                    FP = len(select.difference(truth))
                    FN = len(truth.difference(select))
                    TN = len(all_indices_set.difference(truth).intersection(all_indices_set.difference(select)))


                    # sensitivity = TP / (TP + FN)
                    # specificity = TN / (FP + TN)

                    # print("Sensitivity: ", sensitivity, "Specificity: ", specificity)

                    Type_I_error = FP / (FP + TN) 
                    Power = TP / (TP + FN)
                    print("Type_I_error: ",Type_I_error,"Power: ",Power)

                    tmp_result = [K,n,r,tr,i,Type_I_error,Power,TP,FP,FN,TN]
                    results.append(tmp_result)



column_names = ['k','n','r','tr','i','Type_I_error','Power','TP','FP','FN','TN']
results_df = pd.DataFrame(results, columns=column_names)
results_df.to_csv('./Simulation_2/EM_summary.tsv', sep='\t', index=False)

if mu_sigma_results_K3:
    mu_sigma_df_K3 = pd.DataFrame(mu_sigma_results_K3)
    mu_sigma_df_K3.to_csv('./Simulation_2/EM_mu_sigma_K3.tsv', sep='\t', index=False)
    print("Saved EM_mu_sigma_K3.tsv")
# Save mu and sigma results for K=5
if mu_sigma_results_K5:
    mu_sigma_df_K5 = pd.DataFrame(mu_sigma_results_K5)
    mu_sigma_df_K5.to_csv('./Simulation_2/EM_mu_sigma_K5.tsv', sep='\t', index=False)
    print("Saved EM_mu_sigma_K5.tsv")
