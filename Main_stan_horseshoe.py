import numpy as np
import stan
import arviz as az
import os
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

def GMM(X, Y, K, file_name, doparallel=False, grainsize=1, processes_per_chain=4, kmeans_seed=42, random_seed=42):

    # K-means -> prior
    y_reshaped = Y.reshape(-1, 1)
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=kmeans_seed).fit(y_reshaped)
    mu_init = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    sigma2_init = np.array([np.mean((Y[labels == k] - mu_init[k]) ** 2) for k in range(K)])
    sigma_init = np.sqrt(sigma2_init)
    ## sorting
    sorted_cluster_ind = np.argsort(mu_init)
    sorted_mu_init = mu_init[sorted_cluster_ind]
    sorted_sigma_init = sigma_init[sorted_cluster_ind]

    # prepare the data for stan
    stan_data = {
        'N': X.shape[0],
        'D': X.shape[1],
        'K': K,
        'X': X,
        'y': Y,
        'mu_prior': sorted_mu_init,
        'sigma_prior': sorted_sigma_init,
        'doparallel': int(doparallel),
        'grainsize': grainsize
    }

    print("mu_init: ",sorted_mu_init, flush=True)
    print("sigma_init: ",sorted_sigma_init, flush=True)

    # 读取Stan模型
    with open("./model_horseshoe_kmeans.stan", 'r') as f:
        stan_code = f.read()

    # 编译模型
    posterior = stan.build(stan_code, data=stan_data, random_seed=random_seed)

    # 运行采样
    fit = posterior.sample(
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
        delta=0.95 #0.95
    )

    # 转换为InferenceData
    inference_data = az.from_pystan(posterior=fit)

    # 保存结果
    result_name = file_name.replace("npz","nc")
    print(result_name)
    if os.path.exists(result_name):
        os.remove(result_name)

    print("Label Switching", flush=True)

    # label switching
    mu = np.mean(inference_data.posterior["mu_k"].values, axis=1)
    sigma = np.mean(inference_data.posterior["sigma_k"].values, axis=1)
    coord = np.stack((mu,sigma), axis=-1)

    ref = coord[0]
    sort = np.zeros((4,K))
    sort[0,:] = range(K)

    for i in range(1,4):
        chain = coord[i]
        C = np.linalg.norm(ref[:, None, :] - chain[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(C)
        sort[i] = col_ind

    sort = sort.astype(int)

    for var_name, values in inference_data.posterior.items():
        if values.shape[-1]==K:
            num_dims = values.ndim
            expand_shape = (slice(None),) + (None,) * (num_dims - 2) + (slice(None),)
            sort_expanded = sort[expand_shape]
            inference_data.posterior[var_name].values = np.take_along_axis(inference_data.posterior[var_name].values, sort_expanded, axis=-1)

    inference_data.to_netcdf(result_name)

    return(inference_data)
