import numpy as np
from scipy.stats import invgamma
from bed_reader import open_bed

def simulation_construction(N=1000, D=1000, ratio=1, N_work=0.1, tratio=1, K=3, snp_gene_threshold=0, rng=np.random.default_rng(None), theta=None):

    ## get X
    ## discrete
    N_snps = round(D*ratio/(ratio+1))
    bed = open_bed("/home/sleeper/yybb/HME/Simulation/genomics/sim_SNP.bed")
    X_snp = bed.read().astype(np.int8)
    X_snp -= 1
    X_snp = X_snp[rng.choice(X_snp.shape[0], N, replace=False),:][:,rng.choice(X_snp.shape[1], N_snps, replace=False)]
    ## continous
    N_gene = D - N_snps
    X_mean = np.full(N_gene, 0)
    X_cov  = np.eye(N_gene)
    X_gene = rng.multivariate_normal(X_mean, X_cov, N)
    ## mixed
    X = np.concatenate((X_snp,X_gene), axis=1)

    ## get cluster charateristics
    mu = rng.normal(loc=0, scale=10, size=K) # 10 
    sigma = invgamma.rvs(a=1, scale=1, size=K,random_state=rng)
    # add some constraints:
    while np.any(mu[:-1] - mu[1:] < 4 * np.sqrt(np.maximum(sigma[:-1], sigma[1:]))):
        mu = rng.normal(loc=0, scale=10, size=K)
        sigma = invgamma.rvs(a=1, scale=1, size=K,random_state=rng)

    # mu = np.array(range(-(K//2), K//2 + 1))*5
    # sigma = np.ones(K)

    ## sparse coefficients
    if theta is None:
        theta = rng.uniform(0, N_work)

    w = np.zeros(D, dtype=int)
    # w = np.sort(w)[::-1]

    ## distribute true signals with tratio to modalities
    total_w = round(D*theta)
    snp_w = round(total_w*tratio/(tratio+1))
    gene_w = total_w-snp_w

    snp_indices = rng.choice(N_snps, size=int(snp_w), replace=False)
    w[snp_indices] = 1

    gene_indices = rng.choice(N_gene, size=int(gene_w), replace=False)
    w[N_snps + gene_indices] = 1

    # for noninformative features, they showed the same coefficient with different groups ~ N(0, 1)
    eta_kd = np.zeros((K, D))
    non_inf_index = np.where(w == 0)[0]
    eta_val = rng.normal(0, 1, size=len(non_inf_index))
    eta_kd[:,non_inf_index] = eta_val

    # for informative features, they showed different coefficients with different groups from different uniformed-distributed regions
    inf_index = np.where(w == 1)[0]
    interval_length = 2
    interval_gap = 2
    center_distance = interval_length + interval_gap
    start = -(K - 1) * center_distance / 2
    interval_centers = start + np.arange(K) * center_distance
    interval_lb = interval_centers - interval_length / 2
    interval_ub = interval_centers + interval_length / 2
    uniform_intervals = np.stack([interval_lb, interval_ub], axis=1)

    dist_index = np.zeros((K, len(inf_index)), dtype=int)
    for i in range(len(inf_index)):
        dist_index[:, i] = rng.permutation(K)

    N_lb = interval_lb[dist_index]
    N_up = interval_ub[dist_index]

    eta_val = rng.uniform(N_lb,N_up,size=(K, len(inf_index)))
    eta_kd[:,inf_index] = eta_val

    lin = X@eta_kd.T
    Z = np.argmax(lin, axis=1)
    Y = rng.normal(loc=mu[Z], scale=np.sqrt(sigma[Z]))

    return X, Y, Z, mu, sigma, theta, eta_kd, w