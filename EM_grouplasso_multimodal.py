import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
import random
import optuna
from sklearn.model_selection import KFold


def kmeans_plusplus_init_1d(data, K):
    # random.seed(26)
    n_samples = len(data)
    mu_init = np.zeros(K)
    first_center_idx = np.random.choice(n_samples)
    mu_init[0] = data[first_center_idx]
    distances_sq = (data - mu_init[0]) ** 2
    for k_idx in range(1, K):
        probabilities = distances_sq / np.sum(distances_sq)
        next_center_idx = np.random.choice(n_samples, p=probabilities)
        mu_init[k_idx] = data[next_center_idx]
        new_distances_sq = (data - mu_init[k_idx]) ** 2
        distances_sq = np.minimum(distances_sq, new_distances_sq)
        # print(mu_init)
    return np.sort(mu_init)


def proximal_gradient_descent(
    X,
    r,
    lambda_reg,
    H_init=None,
    max_iter=2000,
    tol=1e-4,
    learning_rate=1e-4,
    verbose=False,
    feature_groups=None,
):
    N, D = X.shape
    K = r.shape[1]
    if H_init is None:
        H = np.zeros((K, D))
    else:
        H = H_init.copy()
        assert H.shape == (K, D), "H_init must have shape (K, D)"

    if isinstance(lambda_reg, dict):
        if feature_groups is None:
            raise ValueError("feature_groups must be provided if lambda_reg is a dict.")
        lambda_vec = np.zeros(D)
        for group_name, indices in feature_groups.items():
            if group_name not in lambda_reg:
                raise ValueError(
                    f"lambda_reg dict is missing penalty for group '{group_name}'"
                )
            lambda_vec[indices] = lambda_reg[group_name]
    elif isinstance(lambda_reg, (int, float)):
        lambda_vec = np.full(D, lambda_reg)
    else:
        raise ValueError("lambda_reg must be a float/int or a dict.")
    rX = r.T @ X
    prev_obj = float("inf")
    for iteration in range(max_iter):
        if verbose:
            print(f"PGD: Iteration {iteration + 1}/{max_iter}")
        logits = X @ H.T
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        gradient = (probs.T @ X - rX) / N
        H_temp = H - learning_rate * gradient
        H_norms = np.linalg.norm(H_temp, axis=0)
        threshold_vec = learning_rate * lambda_vec
        shrinkage_factors = np.maximum(
            1 - threshold_vec / np.maximum(H_norms, 1e-12), 0
        )
        H_new = H_temp * shrinkage_factors[np.newaxis, :]
        logits_new = X @ H_new.T
        log_sum_exp = logsumexp(logits_new, axis=1)
        weighted_logits = np.sum(rX * H_new)
        smooth_obj = -weighted_logits / N + np.sum(log_sum_exp) / N
        H_new_norms = np.linalg.norm(H_new, axis=0)
        group_penalty = np.sum(lambda_vec * H_new_norms)
        obj_value = smooth_obj + group_penalty
        if iteration > 0:
            if verbose:
                print(
                    f"PGD: Iteration {iteration + 1}, Objective: {obj_value:.4f}, Previous Objective: {prev_obj:.4f}"
                )
                if obj_value > prev_obj:
                    print(
                        f"Warning: Objective increased at iteration {iteration + 1}. Please check step size!"
                    )
            rel_change = abs(obj_value - prev_obj) / (abs(prev_obj) + 1e-12)
            if rel_change < tol:
                break
        prev_obj = obj_value
        H = H_new
    return H


def fista_optimization(
    X,
    r,
    lambda_reg,
    H_init=None,
    max_iter=2000,
    tol=1e-4,
    learning_rate=1e-4,
    verbose=False,
    feature_groups=None,
):
    N, D = X.shape
    K = r.shape[1]
    if H_init is None:
        H = np.zeros((K, D))
    else:
        H = H_init.copy()
        assert H.shape == (K, D), "H_init must have shape (K, D)"

    if isinstance(lambda_reg, dict):
        if feature_groups is None:
            raise ValueError("feature_groups must be provided if lambda_reg is a dict.")
        lambda_vec = np.zeros(D)
        for group_name, indices in feature_groups.items():
            if group_name not in lambda_reg:
                raise ValueError(
                    f"lambda_reg dict is missing penalty for group '{group_name}'"
                )
            lambda_vec[indices] = lambda_reg[group_name]
    elif isinstance(lambda_reg, (int, float)):
        lambda_vec = np.full(D, lambda_reg)
    else:
        raise ValueError("lambda_reg must be a float/int or a dict.")
    Y = H.copy()
    t = 1.0
    rX = r.T @ X
    step_size = learning_rate
    beta = 0.5

    def compute_objective(H_eval):
        logits = X @ H_eval.T
        log_sum_exp = logsumexp(logits, axis=1)
        weighted_logits = np.sum(rX * H_eval)
        smooth_part = -weighted_logits / N + np.sum(log_sum_exp) / N
        group_penalty = np.sum(lambda_vec * np.linalg.norm(H_eval, axis=0))
        return smooth_part + group_penalty

    def compute_gradient(H_eval):
        logits = X @ H_eval.T
        probs = np.exp(
            logits - logsumexp(logits, axis=1, keepdims=True)
        )  # Use logsumexp for stability
        return (probs.T @ X - rX) / N

    def proximal_operator(H_temp, step):
        H_new = np.zeros_like(H_temp)
        for d in range(D):
            col_norm = np.linalg.norm(H_temp[:, d])
            threshold = step * lambda_vec[d]
            if col_norm > threshold:
                H_new[:, d] = (1 - threshold / col_norm) * H_temp[:, d]
        return H_new

    prev_obj = compute_objective(H)
    for iteration in range(max_iter):
        if verbose:
            print(f"FISTA: Iteration {iteration + 1}/{max_iter}")
        H_prev = H.copy()
        current_step = step_size
        max_backtrack = 15
        for backtrack in range(max_backtrack):
            if verbose:
                print(f"FISTA: Backtrack {backtrack + 1}/{max_backtrack}")
            try:
                grad = compute_gradient(Y)
                grad_norm = np.linalg.norm(grad)
                if not np.isfinite(grad_norm) or grad_norm > 1e6:
                    current_step *= beta
                    continue
                H_temp = Y - current_step * grad
                H_new = proximal_operator(H_temp, current_step)
                obj_new = compute_objective(H_new)
                if not np.isfinite(obj_new):
                    current_step *= beta
                    continue
                if (
                    obj_new <= prev_obj + 1e-4 * current_step * grad_norm**2
                    or backtrack == max_backtrack - 1
                ):
                    break
                else:
                    current_step *= beta
            except (OverflowError, RuntimeWarning):
                current_step *= beta
                continue
        H = H_new
        try:
            current_obj = compute_objective(H)
            if not np.isfinite(current_obj):
                print(
                    f"Warning: Objective became non-finite at iteration {iteration + 1}"
                )
                step_size = min(step_size, 1e-4)
                H = np.random.normal(0, 0.01, (K, D))
                Y = H.copy()
                t = 1.0
                continue
            if iteration > 0:
                if verbose:
                    print(
                        f"FISTA: Iteration {iteration + 1}, Objective: {prev_obj:.4f} -> {current_obj:.4f}"
                    )
                    if current_obj > prev_obj:
                        print(
                            f"Warning: Objective increased at iteration {iteration + 1}. Please check step size!"
                        )
                rel_change = abs(current_obj - prev_obj) / (abs(prev_obj) + 1e-12)
                if rel_change < tol:
                    if verbose:
                        print(f"FISTA: Converged at iteration {iteration}")
                    break
            prev_obj = current_obj
        except Exception as e:
            print(f"Error computing objective at iteration {iteration}: {e}")
            break
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        momentum_coeff = (t - 1) / t_new
        Y_new = H + momentum_coeff * (H - H_prev)
        if np.any(~np.isfinite(Y_new)) or np.linalg.norm(Y_new) > 1e3:
            Y = H.copy()
            t = 1.0
        else:
            Y = Y_new
            t = t_new
        if iteration % 100 == 0 and iteration > 0:
            step_size = min(step_size * 1.05, 0.1)
    return H


def mstep_with_group_lasso(
    X,
    y,
    r,
    lambda_reg,
    gamma_mu=0.0,
    gamma_sigma=0.0,
    epsilon_mu=1e-6,
    H_init=None,
    opt_method="pgd",
    opt_max_iter=2000,
    opt_tol=1e-4,
    learning_rate=1e-4,
    opt_verbose=False,
    feature_groups=None,
):
    N, D = X.shape
    K = r.shape[1]
    r_sum = np.sum(r, axis=0) + 1e-8
    mu_unconstrained = (r.T @ y) / r_sum
    y_centered = y[:, np.newaxis] - mu_unconstrained[np.newaxis, :]
    sigma2_unconstrained = np.sum(r * y_centered**2, axis=0) / r_sum
    sigma2_unconstrained = np.maximum(sigma2_unconstrained, 1e-8)

    var_y = np.var(y)

    if gamma_mu > 0:

        def mu_objective(mu_vec):
            obj = 0.0
            N_eff = np.sum(r, axis=0)
            for k in range(K):
                obj += np.sum(r[:, k] * (y - mu_vec[k]) ** 2) / (
                    2 * sigma2_unconstrained[k]
                )
            for k in range(K - 1):
                for l in range(k + 1, K):
                    # var_diff_mu = (sigma2_unconstrained[k] / (N_eff[k] + 1e-8)) + \
                    #       (sigma2_unconstrained[l] / (N_eff[l] + 1e-8))
                    # diff_mu_sq = (mu_vec[k] - mu_vec[l])**2
                    # z_score_sq = diff_mu_sq / var_diff_mu
                    # obj += gamma_mu / (z_score_sq + epsilon_mu)
                    obj += gamma_mu / ((mu_vec[k] - mu_vec[l]) ** 2 + epsilon_mu)
            return obj

        def mu_gradient(mu_vec):
            grad = np.zeros(K)
            N_eff = np.sum(r, axis=0)

            for k in range(K):
                grad[k] = -np.sum(r[:, k] * (y - mu_vec[k])) / sigma2_unconstrained[k]
            for k in range(K):
                for l in range(K):
                    if k != l:
                        # var_diff_mu = (sigma2_unconstrained[k] / (N_eff[k] + 1e-8)) + \
                        #   (sigma2_unconstrained[l] / (N_eff[l] + 1e-8))
                        diff = mu_vec[k] - mu_vec[l]
                        # z_score_sq = (diff**2) / var_diff_mu
                        # grad[k] += -2 * gamma_mu * diff / \
                        #    (var_diff_mu * (z_score_sq + epsilon_mu)**2)
                        grad[k] -= 2 * gamma_mu * diff / ((diff**2 + epsilon_mu) ** 2)
            return grad

        result = minimize(
            mu_objective,
            mu_unconstrained,
            method="BFGS",
            jac=mu_gradient,
            options={"gtol": 1e-6},
        )
        mu = result.x
    else:
        mu = mu_unconstrained

    def sigma2_objective(log_sigma2_vec):
        sigma2_vec = np.exp(
            log_sigma2_vec
        )  # Optimize in log-space to ensure positivity

        # Likelihood term (negative log-likelihood part)
        obj_likelihood = 0.0
        for k in range(K):
            obj_likelihood += np.sum(
                r[:, k]
                * (
                    0.5 * np.log(2 * np.pi * sigma2_vec[k])
                    + 0.5 * (y - mu[k]) ** 2 / sigma2_vec[k]
                )
            )

        penalty = 0.0
        for k in range(K):
            # diff_sq = (np.minimum(sigma2_vec[k], var_y) - var_y)**2
            # penalty += gamma_sigma / (diff_sq + 1e-6)
            penalty += gamma_sigma * sigma2_vec[k] ** 2

        return obj_likelihood + penalty

    y_centered = y[:, np.newaxis] - mu[np.newaxis, :]
    sigma2 = np.sum(r * y_centered**2, axis=0) / r_sum
    sigma2 = np.maximum(sigma2, 1e-8)

    if gamma_sigma > 0:
        log_sigma2_initial_guess = np.log(sigma2)
        result = minimize(
            sigma2_objective,
            log_sigma2_initial_guess,
            method="BFGS",
            options={"gtol": 1e-6},
        )

        sigma2 = np.exp(result.x)
        sigma2 = np.maximum(sigma2, 1e-8)

    if opt_method == "pgd":
        H = proximal_gradient_descent(
            X,
            r,
            lambda_reg,
            H_init=H_init,
            max_iter=opt_max_iter,
            tol=opt_tol,
            learning_rate=learning_rate,
            verbose=opt_verbose,
            feature_groups=feature_groups,
        )
    elif opt_method == "fista":
        H = fista_optimization(
            X,
            r,
            lambda_reg,
            H_init=H_init,
            max_iter=opt_max_iter,
            tol=opt_tol,
            learning_rate=learning_rate,
            verbose=opt_verbose,
            feature_groups=feature_groups,
        )
    else:
        raise ValueError("opt_method must be either 'pgd' or 'fista'")
    return H, mu, sigma2


def em_moe_gmm(
    X,
    y,
    K,
    lambda_reg,
    gamma_mu=0.0,
    gamma_sigma=0.0,
    epsilon_mu=1e-6,
    max_iter=100,
    tol=1e-4,
    opt_method="pgd",
    opt_max_iter=2000,
    opt_tol=1e-4,
    learning_rate=1e-4,
    opt_verbose=False,
    init_method="quantile",
    feature_groups=None,
):
    # np.random.seed(26)

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), (
        "X and y must be numpy arrays"
    )
    assert X.ndim == 2, "X must be a 2D array"
    N, D = X.shape
    assert K > 1, "K must be greater than 1"
    assert y.ndim == 1 and y.shape[0] == N, (
        "y must be a 1D array with the same number of samples as X"
    )

    if init_method == "quantile":
        y_min, y_max = np.percentile(y, [10, 90])
        mu = np.linspace(y_min, y_max, K)
    elif init_method == "k_plus":
        mu = kmeans_plusplus_init_1d(y, K)
    else:
        raise ValueError("init_method must be either 'quantile' or 'k_plus'")

    sigma2 = np.ones(K) * np.var(y) * 0.5
    H = np.zeros((K, D))
    prev_ll = float("-inf")

    # print(mu, sigma2)

    for iteration in range(max_iter):
        logits = X @ H.T
        log_pi = logits - logsumexp(logits, axis=1, keepdims=True)
        pi = np.exp(log_pi)
        # print(pi.max(axis=1)[:10])

        y_expanded = y[:, np.newaxis]
        mu_expanded = mu[np.newaxis, :]
        sigma2_expanded = sigma2[np.newaxis, :]

        log_likelihood = (
            -0.5 * np.log(2 * np.pi * sigma2_expanded)
            - 0.5 * (y_expanded - mu_expanded) ** 2 / sigma2_expanded
        )

        log_joint = log_pi + log_likelihood

        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_r = log_joint - log_marginal
        r = np.exp(log_r)
        # print(r.max(axis=1)[:10])

        ll = np.sum(log_marginal)

        if iteration > 0:
            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-12)
            if rel_change < tol:
                print(f"EM algorithm converged after {iteration + 1} iterations")
                break

        prev_ll = ll

        H, mu, sigma2 = mstep_with_group_lasso(
            X,
            y,
            r,
            lambda_reg,
            gamma_mu=gamma_mu,
            gamma_sigma=gamma_sigma,
            epsilon_mu=epsilon_mu,
            H_init=H,
            opt_method=opt_method,
            opt_max_iter=opt_max_iter,
            opt_tol=opt_tol,
            learning_rate=learning_rate,
            opt_verbose=opt_verbose,
            feature_groups=feature_groups,
        )
        # print(mu, sigma2)

    active_set = []
    feature_norms = np.linalg.norm(H, axis=0)
    for d in range(D):
        if feature_norms[d] > 1e-6:
            active_set.append(d)
    return H, mu, sigma2, active_set


def calculate_ll_new(X_new, y_new, H, mu, sigma2):
    N_new, D = X_new.shape
    K = H.shape[0]
    logits = X_new @ H.T
    log_pi = logits - logsumexp(logits, axis=1, keepdims=True)
    y_expanded = y_new[:, np.newaxis]
    mu_expanded = mu[np.newaxis, :]
    sigma2_expanded = sigma2[np.newaxis, :]

    log_expert_likelihood = (
        -0.5 * np.log(2 * np.pi * sigma2_expanded)
        - 0.5 * (y_expanded - mu_expanded) ** 2 / sigma2_expanded
    )

    log_combined = log_pi + log_expert_likelihood
    log_marginal = logsumexp(log_combined, axis=1)
    total_log_likelihood = np.sum(log_marginal)
    return total_log_likelihood
