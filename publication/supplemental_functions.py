#!/usr/bin/python
from __future__ import division
import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import gamma, gammaln, polygamma
from scipy.optimize import minimize_scalar
from math import pi

TINY_FLOAT64 = sp.finfo(sp.float64).tiny


"""
Gaussian Kernel Density Estimation
"""


# Gaussian kernel density estimation with cross validation and bootstrap sampling
def gkde(data0, xs, num_samples=0, num_h=100, massage_J=True, tolerance=1E-3, ERROR_switch=False):

    data = data0.copy()
    N = len(data)
    G = len(xs)
    dx = xs[1] - xs[0]

    # Sort data
    data.sort()

    # Set h_min to minimal data spacing. Shift data if there are ties
    h_min = np.diff(data).min()
    if h_min == 0.:
        # This ensures the shifted data will round to the correct value (to 1st decimal for WHO data)
        data_shifted = np.zeros(N)  # Do not change data directly. Use data_shifted!
        for i in range(N):
            if data[i] == xs.min():
                data_shifted[i] = data[i] + 0.05 * np.random.rand()
            if xs.min() < data[i] < xs.max():
                data_shifted[i] = data[i] + 0.10 * (np.random.rand() - 0.5)
            if data[i] == xs.max():
                data_shifted[i] = data[i] - 0.05 * np.random.rand()
        data = data_shifted
        data.sort()
        h_min = np.diff(data).min()
        # If there are still ties, give up
        if h_min == 0.:
            Q_star, Q_samples, ERROR_switch = None, None, True
            return Q_star, Q_samples, ERROR_switch

    # Set h_max to maximal data spacing x 10
    h_max = (data.max()-data.min()) * 10

    # Form hs
    hs = np.geomspace(h_min, h_max, num_h)

    # For each h, compute the risk function
    Js = np.zeros(num_h)
    for k in range(num_h):
        h = hs[k]
        sum_over_i = 0.
        for i in range(N):
            data_i = list(data.copy())
            data_i.pop(i)
            Q_i = gaussian_kde(data_i, bw_method=h)(xs)
            Q_i /= (sp.sum(Q_i*dx) + TINY_FLOAT64)
            # Set negative interpolated values (occurring when h is very small) to 0
            value = max(float(interp1d(xs, Q_i, kind='cubic', fill_value="extrapolate")(data[i])), 0.)
            sum_over_i += np.log(value + TINY_FLOAT64)
        J = - sum_over_i
        # Terminate if got an nan from gaussian_kde
        if np.isnan(J):
            Q_star, Q_samples, ERROR_switch = None, None, True
            return Q_star, Q_samples, ERROR_switch
        Js[k] = J

    # Massage Js so that the risk function is better-behaved
    if massage_J:
        Js = Js - Js.min() + 1.0
        Js = np.log(Js)

    # Interpolate the risk function
    J_func = interp1d(hs, Js, kind='cubic')

    # Compute 1st derivative of the risk function
    dJdhs = np.gradient(Js)

    # Solve for all hs that correspond to local extrema of the risk function
    hs_solved, Js_solved = [], []
    for k in range(num_h-1):
        if dJdhs[k] * dJdhs[k+1] < 0:
            h_k = h_solver(hs[k], hs[k+1], hs, dJdhs, tolerance)
            J_k = float(J_func(h_k))
            hs_solved.append(h_k)
            Js_solved.append(J_k)

    # Pick up h_star that corresponds to the global minimum of the risk function
    if len(hs_solved) > 0:
        h_star = hs_solved[sp.array(Js_solved).argmin()]
        # If this minimum is actually local, set h_star to either h_max or h_min
        if (min(Js_solved) > Js[0]) or (min(Js_solved) > Js[-1]):
            if Js[0] > Js[-1]:
                h_star = h_max
            elif Js[0] < Js[-1]:
                h_star = h_min
    # If no h were solved, set h_star to either h_max or h_min
    else:
        if Js[0] > Js[-1]:
            h_star = h_max
        elif Js[0] < Js[-1]:
            h_star = h_min

    # Estimate the optimal density with h_star
    Q_star = gaussian_kde(data, bw_method=h_star)(xs)
    Q_star /= sp.sum(Q_star*dx)

    # Use bootstrap to estimate uncertainty (h is fixed at h_star)
    Q_samples = np.zeros([G,num_samples])
    for k in range(num_samples):
        bootstrapped_data = np.random.choice(data, size=N, replace=True)
        Q_k = gaussian_kde(bootstrapped_data, bw_method=h_star)(xs)
        Q_k /= sp.sum(Q_k*dx)
        Q_samples[:,k] = Q_k

    # Return
    return Q_star, Q_samples, ERROR_switch


# Solve h at which dJdh = 0 using bisection
def h_solver(h_lb, h_ub, hs, dJdhs, tolerance):
    h1, h2 = h_lb, h_ub
    hm_old = np.inf
    while True:
        hm = (h1+h2)/2
        if abs(hm-hm_old) < tolerance:
            break
        hm_old = hm
        f1 = dJdh_func(h1, hs, dJdhs)
        f2 = dJdh_func(h2, hs, dJdhs)
        fm = dJdh_func(hm, hs, dJdhs)
        if f1*fm < 0:
            h1, h2 = h1, hm
        elif fm*f2 < 0:
            h1, h2 = hm, h2
    return hm


# 1st derivative of the risk function
def dJdh_func(h, hs, dJdhs):
    return interp1d(hs, dJdhs, kind='cubic')(h)


"""
Dirichlet Process Mixture Modeling
"""


# Dirichlet process mixture modeling with Gibbs sampling
def dpmm(data, xs, num_samples=100, num_thermalization=100, H=10, M=1, ERROR_switch=False):

    N = len(data)
    G = len(xs)

    # Initialize
    kappa = 1
    mu0 = sp.mean(data)
    alpha0 = 1
    beta0 = sp.std(data)**2

    m_array = np.zeros([H,2])
    m_array[:,1] = invgamma_sampler(alpha=alpha0, beta=beta0, size=H)
    for h in range(H):
        m_array[h,0] = np.random.normal(loc=mu0, scale=sp.sqrt(kappa*m_array[h,1]), size=1)

    w_array = np.ones(H) / H

    # Gibbs sampling
    Q_samples = np.zeros([G,num_samples])
    for k in range(num_thermalization+num_samples):

        # Update clustering
        r_array = np.zeros(N)
        for i in range(N):
            wf = np.zeros(H)
            for h in range(H):
                wf[h] = w_array[h] * normal(x=data[i], mu=m_array[h,0], sigma=sp.sqrt(m_array[h,1]))
            wf /= sp.sum(wf)
            r_array[i] = np.random.choice(range(H), size=1, p=wf)

        r_list = [int(r_array[i]) for i in range(N)]

        # Update locations
        m_array = np.zeros([H,2])
        for h in range(H):
            i_list = []
            for i in range(N):
                if r_list[i] == h:
                    i_list.append(i)
            n_h = len(i_list)
            if n_h > 0:
                data_h = data[i_list]
                data_mean_h = sp.mean(data_h)
                kappa_h = 1 / (1/kappa + n_h)
                mu_h = kappa_h * (mu0/kappa + n_h*data_mean_h)
                alpha_h = alpha0 + n_h / 2
                beta_h = beta0 + (sp.sum((data_h-data_mean_h)**2) + n_h/(1+kappa*n_h)*(data_mean_h-mu0)**2) / 2
                m_array[h,1] = invgamma_sampler(alpha=alpha_h, beta=beta_h, size=1)
                m_array[h,0] = np.random.normal(loc=mu_h, scale=sp.sqrt(kappa_h*m_array[h,1]), size=1)
            else:
                m_array[h,1] = invgamma_sampler(alpha=alpha0, beta=beta0, size=1)
                m_array[h,0] = np.random.normal(loc=mu0, scale=sp.sqrt(kappa*m_array[h,1]), size=1)

        # Update weights (stick-breaking algorithm)
        A_array = np.zeros(H)
        for h in range(H):
            A_array[h] = r_list.count(h)
        B_array = np.zeros(H)
        for h in range(H):
            B_array[h] = sp.sum(A_array[h+1:])

        v_array = np.zeros(H)
        for h in range(H):
            v_array[h] = np.random.beta(a=A_array[h]+1, b=B_array[h]+M, size=1)

        u_array = np.ones(H) - v_array

        w_array = np.zeros(H)
        w_array[0] = v_array[0]
        for h in range(1, H-1):
            w_array[h] = v_array[h] * np.cumprod(u_array[:h])[-1]
        w_array[-1] = abs(1-sp.sum(w_array))

        # Save samples after thermalization
        if k > num_thermalization-1:
            Q_samples[:,k-num_thermalization] = combine_normals(xs, w_array, m_array)

    # Compute mean of the samples as the optimal density
    Q_star = Q_samples.mean(axis=1)

    # Return
    return Q_star, Q_samples, ERROR_switch


# Inverse-gamma distribution
def invgamma(x, alpha, beta):
    return beta**alpha * sp.exp(-beta/x) / gamma(alpha) / x**(alpha+1)


# Draw random numbers from inverse-gamma distribution
def invgamma_sampler(alpha, beta, size, invgamma_min=1E-3):
    x_start = beta/(alpha+1)  # mode (most likely value) of invgamma
    x_lb = x_start
    while invgamma(x_lb, alpha, beta) > invgamma_min:
        x_lb /= 10.0
    x_ub = x_start
    while invgamma(x_ub, alpha, beta) > invgamma_min:
        x_ub *= 10.0
    xs = np.linspace(x_lb, x_ub, 10001)
    dx = xs[1] - xs[0]
    xs = np.linspace(x_lb+dx/2, x_ub-dx/2, 10000)
    prob = invgamma(xs, alpha, beta) / sp.sum(invgamma(xs, alpha, beta))
    samples = np.random.choice(xs, size=size, replace=True, p=prob)
    jitter = dx * (np.random.rand(size)-0.5)
    samples += jitter
    return samples


# Normal distribution
def normal(x, mu, sigma):
    return sp.exp(-(x-mu)**2/(2*sigma**2)) / sp.sqrt(2*pi*sigma**2)


# Combine normal distributions
def combine_normals(xs, w_array, m_array):
    H = len(w_array)
    G = len(xs)
    dx = xs[1] - xs[0]
    wf = np.zeros([H,G])
    for h in range(H):
        wf[h,:] = w_array[h] * normal(xs, mu=m_array[h,0], sigma=sp.sqrt(m_array[h,1]))
    Q = wf.sum(axis=0)
    Q /= sp.sum(Q*dx)
    return Q


"""
Some utility functions
"""


# Compute log-likelihood per datum
def likelihood(xs, Q, data):
    Q_func = interp1d(xs, Q, kind='cubic', fill_value="extrapolate")
    L_data = 1/len(data) * sp.sum(sp.log(Q_func(data) + TINY_FLOAT64))
    return L_data


# Compute Kullback-Leibler divergence, D_KL(P||Q)
def KL_divergence(P, Q, dx):
    D_KL = sp.sum(dx * P * sp.log((P+TINY_FLOAT64)/(Q+TINY_FLOAT64)))
    return D_KL


# Given a set of data, compute p-value of an arbitrary data point
def p_value_cal(data, point):
    count = 0
    for i in range(len(data)):
        if data[i] <= point:
            count += 1
    p_value = count/len(data)
    return p_value


"""
Entropy Estimators
"""


# Naive estimator. Ref: Justin's dissertation
def naive_estimator(data, N, G, bbox):

    # Make a histogram of the data and get the count in each bin
    bin_edges = np.linspace(bbox[0], bbox[1], G+1)
    counts, bin_edges = np.histogram(a=data, bins=bin_edges)

    # Turn counts into frequencies
    freqs = counts/N

    # Compute entropy, Eqn.(3.15)
    H = -sp.sum(freqs * sp.log(freqs+TINY_FLOAT64))

    # Correct entropy by adding log(L/G)
    L = bbox[1] - bbox[0]
    H += sp.log(L/G)

    # Convert from nats to bits
    H *= sp.log2(sp.exp(1))

    # Return
    return H


# kNN estimator. Ref: A. Kraskov et al, Phys. Rev. E 69, 066138 (2004)
def kNN_estimator(data, N, k):

    # Compute pair-distances between the data points
    pair_dists = abs(sp.array(sp.mat(data).T * sp.mat(np.ones(N)) - sp.mat(np.ones(N)).T * sp.mat(data)))

    # Sort pair-distances, from small to large, for each row
    pair_dists.sort(axis=1)

    # Choose the kNN pair-distances
    kNN_pair_dist = pair_dists[:,k]

    # Compute entropy, Eqn.(20)
    H = polygamma(0,N) - polygamma(0,k) + 1/N * sp.sum(sp.log(2*kNN_pair_dist+TINY_FLOAT64))

    # Convert from nats to bits
    H *= sp.log2(sp.exp(1))

    # Return
    return H


# NSB estimator. Ref: Justin's dissertation
def NSB_estimator(data, N, G, bbox):

    # Make a histogram of the data and get the count in each bin
    bin_edges = np.linspace(bbox[0], bbox[1], G+1)
    counts, bin_edges = np.histogram(a=data, bins=bin_edges)

    # Determine the maximum of the log probability
    beta_star = minimize_scalar(neg_log_prob, method='golden', bounds=(0, np.inf), args=(G, N, counts)).x
    log_prob_beta_star = log_prob(beta_star, G, N, counts)

    # Compute entropy and its variance, Eqn.(3.29) and Eqn.(3.33)
    denom = quad(integrand_p, 0, np.inf, args=(G, N, counts, log_prob_beta_star))[0]
    numer_H = quad(integrand_pH, 0, np.inf, args=(G, N, counts, log_prob_beta_star))[0]
    numer_Hsq = quad(integrand_pHsq, 0, np.inf, args=(G, N, counts, log_prob_beta_star))[0]
    numer_varH = quad(integrand_pvarH, 0, np.inf, args=(G, N, counts, log_prob_beta_star))[0]

    H_mean = numer_H/denom
    H_sq_mean = numer_Hsq/denom
    H_var = numer_varH/denom + H_sq_mean - H_mean**2

    # Correct H mean by adding log(L/G)
    L = bbox[1] - bbox[0]
    H_mean += sp.log(L/G)

    # Convert from nats to bits
    H_mean *= sp.log2(sp.exp(1))
    H_error = np.sqrt(H_var) * sp.log2(sp.exp(1))

    # Return
    return H_mean, H_error


# log of Eqn.(3.32)
def log_prob(beta, G, N, counts):
    if beta <= 0:
        return -np.inf
    else:
        return gammaln(beta*G) - G*gammaln(beta) + sp.sum(gammaln(counts+beta)) - gammaln(N+beta*G) + sp.log(G*polygamma(1,beta*G+1) - polygamma(1,beta+1))


# Negative of log_prob
def neg_log_prob(beta, G, N, counts):
    return -log_prob(beta, G, N, counts)


# Eqn.(3.22)
def H(beta, G, N, counts):
    A = counts + beta + 1
    B = N + beta*G + 1
    return polygamma(0,B) - sp.sum((A-1)/(B-1)*polygamma(0,A))


# Eqn.(3.24)
def var_H(beta, G, N, counts):
    A = counts + beta + 1
    B = N + beta*G + 1
    return sp.sum(A/B*(A-1)/(B-1)*polygamma(1,A)) - polygamma(1,B) + sp.sum(1/B*(A-1)/(B-1)*polygamma(0,A)**2) - 1/B*sp.sum((A-1)/(B-1)*polygamma(0,A))**2


def integrand_p(beta, G, N, counts, log_prob_beta_star):
    return np.exp(log_prob(beta, G, N, counts)-log_prob_beta_star)


def integrand_pH(beta, G, N, counts, log_prob_beta_star):
    return np.exp(log_prob(beta, G, N, counts)-log_prob_beta_star) * H(beta, G, N, counts)


def integrand_pHsq(beta, G, N, counts, log_prob_beta_star):
    return np.exp(log_prob(beta, G, N, counts)-log_prob_beta_star) * H(beta, G, N, counts)**2


def integrand_pvarH(beta, G, N, counts, log_prob_beta_star):
    return np.exp(log_prob(beta, G, N, counts)-log_prob_beta_star) * var_H(beta, G, N, counts)
