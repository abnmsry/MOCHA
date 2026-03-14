import numpy as np
import os
from collections import Counter
import generation_rng
import matplotlib.pyplot as plt


# This one is fit for Simulation 2, where i is iterated for independent datasets generations. However, you can comment this to replicate Simulation 1.

K = 3
N = 500

for r in [1, 5, 10]:
    for tr in [1, 5]:
        result_dir = f"./Simulation/Simulation_2/k{K}n{N}r{r}tr{tr}/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if (K == 3) and (N == 200):
            set_size_min = 50
        elif (K == 5) and (N == 200):
            set_size_min = 20
        elif (K == 3) and (N == 500):
            set_size_min = 120
        elif (K == 5) and (N == 500):
            set_size_min = 70

        D = 500
        seed = 1
        theta = 0.02
        N_large = N * K 
        print(f"Starting K={K}, N={N}, r={r}, tr={tr}...")
        
        for i in range(100):

            counter={1:0}
            
            while True:

                rng = np.random.default_rng(seed)

                X_large, Y_large, Z_large, mu, sigma, theta_val, eta_kd, w = generation_rng.simulation_construction(
                    N=N_large, D=D, ratio=r, tratio=tr, K=K, rng=rng, theta=theta)

                counter_large = Counter(Z_large)

                cluster_sizes = [set_size_min] * K
                remaining_N = N - sum(cluster_sizes)

                if (min(counter_large.values()) >= set_size_min) and (len(counter_large) == K):

                    if remaining_N > 0:

                        split_points = np.sort(rng.choice(remaining_N + 1, K - 1, replace=True))
                        splits = np.diff(np.concatenate(([0], split_points, [remaining_N])))
                        cluster_sizes = [cluster_sizes[j] + splits[j] for j in range(K)]
                        
                        if any(cluster_sizes[k] > counter_large[k] for k in range(K)):
                            adjusted_sizes = list(cluster_sizes)
                            exceeding_clusters = [k_val for k_val in range(K) if adjusted_sizes[k_val] > counter_large[k_val]]
                            sorted_keys = sorted(counter_large.keys(), key=lambda k: counter_large[k])
                            adjusted_sizes = sorted(cluster_sizes)
                            adjusted_sizes = [adjusted_sizes[i] for i in sorted_keys]

                            if any(adjusted_sizes[k] > counter_large[k] for k in range(K)):
                                seed+=1
                                continue
                            else:
                                cluster_sizes = adjusted_sizes
                            
                    sampled_X_list = []
                    sampled_Y_list = []
                    sampled_Z_list = []
                    all_sampled_indices_from_large = []

                    for k_val in range(K):
                        target_cluster_size = cluster_sizes[k_val]
                    
                        cluster_indices_in_large = np.where(Z_large == k_val)[0]
                        
                        sampled_indices_for_this_cluster = rng.choice(
                            cluster_indices_in_large, size=target_cluster_size, replace=False)
                        
                        sampled_X_list.append(X_large[sampled_indices_for_this_cluster,:])
                        sampled_Y_list.append(Y_large[sampled_indices_for_this_cluster])
                        sampled_Z_list.append(Z_large[sampled_indices_for_this_cluster])
                
                        all_sampled_indices_from_large.extend(sampled_indices_for_this_cluster)

                    X = np.concatenate(sampled_X_list)
                    Y = np.concatenate(sampled_Y_list)
                    Z = np.concatenate(sampled_Z_list)

                    counter = Counter(Z)

                    print(f"  K={K}, N={N}, r={r}, tr={tr}, Experiment {i}, seed {seed}: {dict(counter)}",flush=True)
                    np.savez_compressed(
                        result_dir + f'dat_{i}.npz',
                        X=X, Y=Y, Z=Z, eta=eta_kd, mu=mu, sigma=sigma, theta=theta_val, w=w, seed=seed)
                    
                    seed+=1
                    
                    break
                    
                else:

                    seed+=1
                
            
print(f"Finished K={K}, N={N}, Final seed={seed}",flush=True)