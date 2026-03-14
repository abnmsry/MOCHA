import numpy as np
import pandas as pd
import os
import EM_grouplasso_multimodal as EM_groupLasso
import optuna
import nest_asyncio

nest_asyncio.apply()

from sklearn.model_selection import KFold


def objective(trial, X, Y, K, feature_groups, kf, em_fixed_params):
    """
    Optuna的目标函数，用于评估一组超参数。
    trial: Optuna Trial object for suggesting hyperparameters
    X_train, y_train: Training data
    X_val, y_val: Validation data for model performance evaluation
    K, feature_groups, em_fixed_params: Fixed parameters passed to em_moe_gmm()
    """

    lambda_genotype = trial.suggest_float("lambda_genotype", 1e-4, 1, log=True)
    lambda_mrna = trial.suggest_float("lambda_mrna", 1e-4, 1, log=True)

    current_lambda_reg_params = {"genotype": lambda_genotype, "mrna": lambda_mrna}

    fold_val_lls = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = Y[train_idx], Y[val_idx]
        try:
            H, mu, sigma2, active_set = EM_groupLasso.em_moe_gmm(
                X_train_fold,
                y_train_fold,
                K,
                lambda_reg=current_lambda_reg_params,
                feature_groups=feature_groups,
                **em_fixed_params,
            )

            val_ll = EM_groupLasso.calculate_ll_new(
                X_val_fold, y_val_fold, H, mu, sigma2
            )
            fold_val_lls.append(val_ll)
        except Exception as e:
            print(f"Trial {trial.number}, Fold {fold}: Error during evaluation: {e}")

            return -np.inf

    avg_val_ll = np.mean(fold_val_lls)

    return avg_val_ll



K = 3
N = 200
r = 1

for tr in [1,5]:

    dat_dir = f"./Simulation/Simulation_2/k{K}n{N}r{r}tr{tr}/"
    result_dir = f"./Simulation_results/Simulation_2/EM_results/k{K}n{N}r{r}tr{tr}"
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i in range(100):

        file_path = result_dir + f"/dat_{i}.npz"

        dat_name = f"dat_{i}.npz"
        sim_dat = np.load(dat_dir + dat_name)

        global_seed = 26

        X = sim_dat["X"]

        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        Y = sim_dat["Y"]

        kf = KFold(n_splits=5, shuffle=True, random_state=113)

        D_total = 500
        D_genotype = round(D_total * r / (r + 1))
        print(D_genotype)

        print(f"i = {i}")

        print("Data loading finished", flush=True)
        np.random.seed(global_seed)

        feature_groups_dict = {
            "genotype": np.arange(D_genotype),
            "mrna": np.arange(D_genotype, D_total),
        }

        em_fixed_params = {
            "gamma_mu": 10,
            "gamma_sigma": 1,
            "epsilon_mu": 1e-6,
            "max_iter": 200,  # maximum iterations for EM
            "tol": 1e-6,  # covergence tolerance for EM
            "opt_method": "fista",  # optimazation for M-step, either "pgd" or "fista"
            "opt_max_iter": 20,  # maximum iterations for M-step optimization
            "opt_tol": 1e-8,  # covergence tolerance for M-step
            "learning_rate": 1,  # learning-rate for M-step optimization
            "opt_verbose": False,  # whether print the optimization process for M-step
            "init_method": "k_plus",  # initialization for mu
        }

        sampler = optuna.samplers.TPESampler(seed=619)

        study = optuna.create_study(direction='maximize', sampler=sampler)

        study.optimize(
            lambda trial: objective(
                trial, X, Y, K, feature_groups_dict, kf, em_fixed_params
            ),
            n_trials=50,
            # timeout=3600
        )

        print("\nOptuna Optimization Finished.")
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best validation Log-Likelihood: {study.best_trial.value:.4f}")
        print(f"Best hyperparameters: {study.best_trial.params}")

        best_lambda_reg = study.best_trial.params
        
        H, mu, sigma2, active_set = EM_groupLasso.em_moe_gmm(
            X,
            Y,
            K,
            lambda_reg={
                "genotype": best_lambda_reg["lambda_genotype"],
                "mrna": best_lambda_reg["lambda_mrna"],
            },
            feature_groups=feature_groups_dict,
            **em_fixed_params,
        )

        print(f"set-size: {len(active_set)}", flush=True)

        np.savez_compressed(
            result_dir + f"/dat_{i}.npz",
            H=H,
            mu=mu,
            sigma=sigma2,
            select=active_set,
            best_lamb=best_lambda_reg,
            log_likelihood=study.best_trial.value,
        )
