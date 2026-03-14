import os
import numpy as np
import pandas as pd

# import sys
import time
import Main_stan_horseshoe
# import argparse

def run_gmm(K):
    start_time = time.time()
    result_file = os.path.join(result_dir, f"dat_K{K}.npz")
    result = Main_stan_horseshoe.GMM(X, Y, K, result_file, doparallel=False)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"K = {K}, runtime: {runtime:.2f} seconds")
    # return K, runtime

if __name__ == "__main__":
    
    # Define directories
    dat_dir = "./Simulation/Simulation_1"
    result_dir = "./Simulation_results/Simulation_1/MCMC_results/K3N200t1tr1"
    dat_name = "dat_K3N200r1tr1.npz"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Load data
    sim_dat = np.load(dat_dir + dat_name)
    X = sim_dat["X"]
    
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    Y = sim_dat["Y"]
    
    # Run GMM for the specified K value
    for K in range(2,7):
        print(f"Running GMM for K = {K}")
        run_gmm(K)
