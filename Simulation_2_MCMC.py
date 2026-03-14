import os
import numpy as np
import pandas as pd

import time
import Main_stan_horseshoe

def run_gmm(K, i):
    start_time = time.time()
    result_file = os.path.join(result_dir, f"dat_{i}.npz")
    result = Main_stan_horseshoe.GMM(X, Y, K, result_file, doparallel=False)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"K = {K}, runtime: {runtime:.2f} seconds")
    # return K, runtime

if __name__ == "__main__":

    K = 5
    N = 500
    r = 10
    
    tr = 1

    # Define directories
    dat_dir = f"./Simulation/Simulation_2/k{K}n{N}r{r}tr{tr}/"
    result_dir = f"./Simulation_results/Simulation_2/MCMC_results/k{K}n{N}r{r}tr{tr}"

    if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    for i in range(100):

        file_path = result_dir + f"/dat_{i}.nc"

        if os.path.exists(file_path):
            continue

        dat_name = f"dat_{i}.npz"

        # Load data
        sim_dat = np.load(dat_dir + dat_name)
        X = sim_dat["X"]
        
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        Y = sim_dat["Y"]
        
        print(f"Running GMM for i = {i}")
        run_gmm(K, i)
