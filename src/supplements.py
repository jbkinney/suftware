from __future__ import division
import numpy as np
import scipy as sp
from scipy.sparse import diags
import multiprocessing as mp
import itertools
import time
import sys

from suftware.src import deft_core
from suftware.src import maxent
from suftware.src import utils
from suftware.src.utils import ControlledError

x_MIN = -500


# Laplace approach with importance sampling
def Laplace_approach(phi_t, R, Delta, t, N, num_samples, go_parallel,
                     pt_sampling=False):
    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N / G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
        # Below are what will be used
        y_dim = alpha
        eig_vals = np.abs(sp.array(U_mat_on_kernel[0]))
        transf_matrix = sp.mat(kernel_basis) * U_mat_on_kernel[1]
        lambdas = sp.exp(-phi_t) * (N / G)
    else:
        G = len(phi_t)
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        # H = deft_code.deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        U_mat = np.linalg.eigh(A_mat)
        # Below are what will be used
        y_dim = G
        eig_vals = np.abs(sp.array(U_mat[0]))
        transf_matrix = U_mat[1]
        lambdas = sp.exp(-phi_t) * (N / G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, draw y samples according to the distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), eig_vals)
        outputs = pool.map(y_sampling_of_Lap, inputs)
        y_samples = sp.array(outputs)
    else:
        y_samples = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            inputs = [num_samples, eig_vals[i]]
            outputs = y_sampling_of_Lap(inputs)
            y_samples[i, :] = outputs

    # Transform y samples to x samples
    x_samples = sp.array(transf_matrix * sp.mat(y_samples))
    for i in range(G):
        x_vec = x_samples[i, :]
        x_vec[x_vec < x_MIN] = x_MIN

    # Shift x samples to get phi samples
    phi_samples = np.zeros([G, num_samples])
    for k in range(num_samples):
        phi_samples[:, k] = x_samples[:, k] + phi_t

    # Calculate the weight of each sample
    x_combo = sp.exp(-x_samples) - np.ones(
        [G, num_samples]) + x_samples - 0.5 * np.square(x_samples)
    dS_vals = sp.array(sp.mat(lambdas) * sp.mat(x_combo)).ravel()
    phi_weights = sp.exp(-dS_vals)

    # If called from posterior sampling, return phi samples along with their weights at this point
    if pt_sampling:
        return phi_samples, phi_weights

    # Calculate sample mean and sample mean std
    w_sample_mean = sp.mean(phi_weights)
    w_sample_mean_std = sp.std(phi_weights) / sp.sqrt(num_samples)

    # Return correction and other stuff
    log_Z_correction = sp.log(w_sample_mean)
    return log_Z_correction, w_sample_mean, w_sample_mean_std


# For each eigen-component, draw y samples according to the distribution
def y_sampling_of_Lap(input_array):
    num_samples = input_array[0]
    eig_val = input_array[1]

    # Draw y samples
    sigma = 1.0 / sp.sqrt(eig_val)
    y_samples = np.random.normal(0, sigma, num_samples)

    # Return y samples
    return y_samples


# Generalized Laplace approach with importance sampling
def GLaplace_approach(phi_t, R, Delta, t, N, num_samples, go_parallel,
                      sampling=True, pt_sampling=False, num_grid=400):
    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N / G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
        # Below are what will be used
        y_dim = alpha
        eig_vals = np.abs(sp.array(U_mat_on_kernel[0]))
        eig_vecs = sp.array((sp.mat(kernel_basis) * U_mat_on_kernel[1]).T)
        transf_matrix = sp.mat(kernel_basis) * U_mat_on_kernel[1]
        lambdas = sp.exp(-phi_t) * (N / G)
    else:
        G = len(phi_t)
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        U_mat = np.linalg.eigh(A_mat)
        # Below are what will be used
        y_dim = G
        eig_vals = np.abs(sp.array(U_mat[0]))
        eig_vecs = sp.array(U_mat[1].T)
        transf_matrix = U_mat[1]
        lambdas = sp.exp(-phi_t) * (N / G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, calculate gamma and draw y samples according to the distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples),
                                itertools.repeat(num_grid),
                                eig_vals, eig_vecs, itertools.repeat(lambdas),
                                itertools.repeat(sampling))
        outputs = pool.map(y_sampling_of_GLap, inputs)
        gammas = np.zeros(y_dim)
        y_samples = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            gammas[i] = outputs[i][0]
            if sampling:
                y_samples[i, :] = outputs[i][1]
    else:
        gammas = np.zeros(y_dim)
        y_samples = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            inputs = [num_samples, num_grid, eig_vals[i], eig_vecs[i, :],
                      lambdas, sampling]
            outputs = y_sampling_of_GLap(inputs)
            gammas[i] = outputs[0]
            if sampling:
                y_samples[i, :] = outputs[1]

    # If not sampling, return correction and other stuff at this point
    if not sampling:
        correction = sp.sum(sp.log(gammas))
        w_sample_mean = 1.0
        w_sample_mean_std = 0.0
        return correction, w_sample_mean, w_sample_mean_std

    # Transform y samples to x samples
    x_samples = sp.array(transf_matrix * sp.mat(y_samples))
    for i in range(G):
        x_vec = x_samples[i, :]
        x_vec[x_vec < x_MIN] = x_MIN

    # Shift x samples to get phi samples
    phi_samples = np.zeros([G, num_samples])
    for k in range(num_samples):
        phi_samples[:, k] = x_samples[:, k] + phi_t

    # Calculate the weight of each sample
    x_combo = sp.exp(-x_samples) - np.ones(
        [G, num_samples]) + x_samples - 0.5 * np.square(x_samples)
    dS_vals = sp.array(sp.mat(lambdas) * sp.mat(x_combo)).ravel()
    if go_parallel:
        inputs = itertools.izip(sp.array(transf_matrix.T), y_samples,
                                itertools.repeat(lambdas))
        outputs = pool.map(dSi_evaluations_of_GLap, inputs)
        dSi_vals = sp.array(outputs)
    else:
        dSi_vals = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            inputs = [sp.array(transf_matrix)[:, i], y_samples[i, :], lambdas]
            outputs = dSi_evaluations_of_GLap(inputs)
            dSi_vals[i, :] = outputs
    sum_dSi_vals = sp.array(sp.mat(np.ones(y_dim)) * sp.mat(dSi_vals)).ravel()
    dS_residues = dS_vals - sum_dSi_vals
    dS_residues[dS_residues < x_MIN] = x_MIN
    phi_weights = sp.exp(-dS_residues)

    # If called from posterior sampling, return phi samples along with their weights at this point
    if pt_sampling:
        return phi_samples, phi_weights

    # Calculate sample mean, sample mean std, and effective sample size of the weights
    w_sample_mean = sp.mean(phi_weights)
    w_sample_mean_std = sp.std(phi_weights) / sp.sqrt(num_samples)

    # Return correction and other stuff
    log_Z_correction = sp.sum(sp.log(gammas)) + sp.log(w_sample_mean)
    return log_Z_correction, w_sample_mean, w_sample_mean_std


# For each eigen-component, calculate gamma and draw y samples according to the distribution
def y_sampling_of_GLap(inputs_array):
    num_samples = inputs_array[0]
    num_grid = inputs_array[1]
    eig_val = inputs_array[2]
    eig_vec = inputs_array[3]
    lambdas = inputs_array[4]
    sampling = inputs_array[5]

    # Find the lower and upper bounds of the Laplace distribution and tabulate its values
    sigma = 1.0 / sp.sqrt(eig_val)
    Lap_N_lb = 0
    while distribution(eig_val, eig_vec, Lap_N_lb * sigma, lambdas,
                       switch=0) > 1E-6:
        Lap_N_lb -= 1
    Lap_N_ub = 0
    while distribution(eig_val, eig_vec, Lap_N_ub * sigma, lambdas,
                       switch=0) > 1E-6:
        Lap_N_ub += 1
    Lap_Ns = []
    Lap_Es = []
    for Lap_N in range(Lap_N_lb, Lap_N_ub + 1):
        Lap_Ns.append(Lap_N)
        Lap_Es.append(
            distribution(eig_val, eig_vec, Lap_N * sigma, lambdas, switch=0))

    # Find the lower and upper bounds of the generalized Laplace distribution and tabulate its values
    sigma = 1.0 / sp.sqrt(eig_val)
    GLap_N_lb = 0
    while distribution(eig_val, eig_vec, GLap_N_lb * sigma, lambdas,
                       switch=1) > 1E-6:
        GLap_N_lb -= 1
    GLap_N_ub = 0
    while distribution(eig_val, eig_vec, GLap_N_ub * sigma, lambdas,
                       switch=1) > 1E-6:
        GLap_N_ub += 1
    GLap_Ns = []
    GLap_Es = []
    for GLap_N in range(GLap_N_lb, GLap_N_ub + 1):
        GLap_Ns.append(GLap_N)
        GLap_Es.append(
            distribution(eig_val, eig_vec, GLap_N * sigma, lambdas, switch=1))

    # See if these two distributions are similar enough
    if Lap_Ns == GLap_Ns:
        diff_Es = abs(sp.array(Lap_Es) - sp.array(GLap_Es))
        if all(diff_Es < 1E-6):
            similar_enough = True
        else:
            similar_enough = False
    else:
        similar_enough = False

    # If these two distributions are similar enough, set gamma to 1, otherwise do the integration
    if similar_enough:
        gamma = 1.0
    else:
        # Evaluate area under the Laplace distribution
        Lap_bin_edges = sp.linspace(Lap_Ns[0] * sigma, Lap_Ns[-1] * sigma,
                                    num_grid + 1)
        h = Lap_bin_edges[1] - Lap_bin_edges[0]
        Lap_bin_centers = sp.linspace(Lap_Ns[0] * sigma + h / 2,
                                      Lap_Ns[-1] * sigma - h / 2, num_grid)
        Lap_bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            Lap_bin_centers_dist[j] = distribution(eig_val, eig_vec,
                                                   Lap_bin_centers[j], lambdas,
                                                   switch=0)
        Lap_area = h * sp.sum(Lap_bin_centers_dist)
        # Evaluate area under the generalized Laplace distribution
        GLap_bin_edges = sp.linspace(GLap_Ns[0] * sigma, GLap_Ns[-1] * sigma,
                                     num_grid + 1)
        h = GLap_bin_edges[1] - GLap_bin_edges[0]
        GLap_bin_centers = sp.linspace(GLap_Ns[0] * sigma + h / 2,
                                       GLap_Ns[-1] * sigma - h / 2, num_grid)
        GLap_bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            GLap_bin_centers_dist[j] = distribution(eig_val, eig_vec,
                                                    GLap_bin_centers[j],
                                                    lambdas, switch=1)
        GLap_area = h * sp.sum(GLap_bin_centers_dist)
        # Take ratio of the areas
        gamma = GLap_area / Lap_area

    # If not sampling, return gamma at this point
    if not sampling:
        return [gamma]

    # If the distribution is normal, draw samples from it directly
    if similar_enough:
        y_samples = np.random.normal(0, sigma, num_samples)

    # Otherwise, draw samples according to the distribution as follows
    else:
        bin_edges = sp.linspace(GLap_N_lb * sigma, GLap_N_ub * sigma,
                                num_grid + 1)
        h = bin_edges[1] - bin_edges[0]
        bin_centers = sp.linspace(GLap_N_lb * sigma + h / 2,
                                  GLap_N_ub * sigma - h / 2, num_grid)
        bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            bin_centers_dist[j] = distribution(eig_val, eig_vec, bin_centers[j],
                                               lambdas, switch=1)
        prob = bin_centers_dist / sp.sum(bin_centers_dist)
        y_samples = np.random.choice(bin_centers, num_samples, replace=True,
                                     p=prob)
        y_shifts = (np.random.random(num_samples) - 0.5 * np.ones(
            num_samples)) * h
        y_samples += y_shifts
        """
            Below is a HOT spot !!!

            # Randomly distribute the samples within each bin
            indices = (y_samples-h/2-N_lb*sigma) / h
            for k in range(num_samples):
                index = int(indices[k])
                a = y_grid[index]
                fa = grid_dist[index]
                fb = grid_dist[index+1]
                r = np.random.rand()
                if fa == fb:
                    y_samples[k] = a + h * r
                else:
                    h_ratio = (sp.sqrt(fa**2+r*(fb**2-fa**2)) - fa) / (fb - fa)
                    y_samples[k] = a + h * h_ratio
            """

    # Return gamma and y samples
    return [gamma, y_samples]


# Evaluations of dSi
def dSi_evaluations_of_GLap(inputs_array):
    Ui = inputs_array[0]
    yi = inputs_array[1]
    lambdas = inputs_array[2]

    G = len(Ui)
    num_samples = len(yi)

    xi = sp.array(sp.mat(Ui).T * sp.mat(yi))
    for i in range(G):
        xi_vec = xi[i, :]
        xi_vec[xi_vec < x_MIN] = x_MIN
    xi_combo = sp.exp(-xi) - np.ones([G, num_samples]) + xi - 0.5 * np.square(
        xi)

    return sp.array(sp.mat(lambdas) * sp.mat(xi_combo)).ravel()


# The Laplace or generalized Laplace distribution
def distribution(eig_val, eig_vec, y, lambdas, switch):
    return sp.exp(
        -(0.5 * eig_val * y ** 2 + switch * dSi(eig_vec * y, lambdas)))


# The dSi function
def dSi(x, lambdas):
    x[x < x_MIN] = x_MIN
    return sp.sum(lambdas * (sp.exp(-x) - 1.0 + x - 0.5 * x ** 2))


# Feynman diagram calculations
def Feynman_diagrams(phi_t, R, Delta, t, N):
    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        # Evaluate propagator matrix
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N / G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        M_inv_on_kernel = sp.linalg.inv(M_mat_on_kernel)
        P_mat = sp.mat(kernel_basis) * M_inv_on_kernel * sp.mat(kernel_basis).T
        # Evaluate vertex vector
        V = sp.exp(-phi_t) * (N / G)
    else:
        G = len(phi_t)
        # Evaluate propagator matrix
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        P_mat = np.linalg.inv(A_mat)
        # Evaluate vertex vector
        V = sp.exp(-phi_t) * (N / G)

    # Calculate Feynman diagrams
    log_Z_correction = diagrams_1st_order(G, P_mat, V)

    # Return the correction and other stuff
    w_sample_mean = 1.0
    w_sample_mean_std = 0.0
    return log_Z_correction, w_sample_mean, w_sample_mean_std


# Feynman diagrams of order 1/N
def diagrams_1st_order(G, P, V):
    s = np.zeros(4)  # s[0] is dummy

    P_diag = sp.array([P[i, i] for i in range(G)])

    # Diagram 1
    s[1] = sp.sum(V * P_diag ** 2)
    s[1] *= -1 / 8

    # Diagram 2
    U = sp.array([V[i] * P_diag[i] for i in range(G)])
    s[2] = sp.array(sp.mat(U) * P * sp.mat(U).T).ravel()[0]
    s[2] *= 1 / 8

    # Diagram 3
    s[3] = sp.array(sp.mat(V) * sp.mat(sp.array(P) ** 3) * sp.mat(V).T).ravel()[
        0]
    s[3] *= 1 / 12

    """
    # Diagram 1
    for i in range(G):
        s[1] += V[i] * P[i,i]**2
    s[1] *= -1/8

    # Diagram 2
    for i in range(G):
        for j in range(G):
            s[2] += V[i] * V[j] * P[i,i] * P[i,j] * P[j,j]
    s[2] *= 1/8

    # Diagram 3
    for i in range(G):
        for j in range(G):
            s[3] += V[i] * V[j] * P[i,j]**3
    s[3] *= 1/12
    """

    # Return
    return sp.sum(s)


# Feynman diagrams of order 1/N^2 ---> Under construction
def diagrams_2nd_order(G, P, V):
    return np.nan


# Metropolis Monte Carlo
def Metropolis_Monte_Carlo(phi_t, R, Delta, t, N, num_samples, go_parallel,
                           pt_sampling):
    G = len(phi_t)
    num_thermalization_steps = 10 * G
    num_steps_per_sample = G

    phi_samples = np.zeros([G, num_samples])
    sample_index = 0

    # Prepare the stuff for the case of maxent or finite t, and then do Monte Carlo sampling
    if not np.isfinite(t):

        # Find the kernel basis
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()

        # Find coefficients of phi_t in the kernel basis
        coeffs = np.zeros(alpha)
        for i in range(alpha):
            coeffs[i] = sp.mat(kernel_basis[:, i]) * sp.mat(phi_t).T

        # Find eigen-modes of the Hessian matrix
        H = maxent.hessian_per_datum_from_coeffs(coeffs, R, kernel_basis)
        A_mat = sp.mat(H) * N
        U_mat = np.linalg.eigh(A_mat)
        eig_vals = np.abs(sp.array(U_mat[0]))
        eig_vecs = np.abs(sp.array(U_mat[1]))

        # Initialize
        coeffs_current = coeffs
        S_current = maxent.action_per_datum_from_coeffs(coeffs_current, R,
                                                        kernel_basis) * N

        # Do Monte Carlo sampling
        for k in range(
                                num_thermalization_steps + num_samples * num_steps_per_sample + 1):
            i = np.random.randint(0, alpha)
            eig_val = eig_vals[i]
            eig_vec = eig_vecs[i, :]
            step_size = np.random.normal(0, 1.0 / np.sqrt(eig_val))
            coeffs_new = coeffs_current + eig_vec * step_size
            S_new = maxent.action_per_datum_from_coeffs(coeffs_new, R,
                                                        kernel_basis) * N
            if np.log(np.random.uniform(0, 1)) < (S_current - S_new):
                coeffs_current = coeffs_new
                S_current = S_new
            if (k > num_thermalization_steps) and (
                    k % num_steps_per_sample == 0):
                phi_samples[:, sample_index] = maxent.coeffs_to_field(
                    coeffs_current, kernel_basis)
                sample_index += 1

    else:

        # Find eigen-modes of the Hessian matrix
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        U_mat = np.linalg.eigh(A_mat)
        eig_vals = np.abs(sp.array(U_mat[0]))
        eig_vecs = np.abs(sp.array(U_mat[1]))

        # Initialize
        phi_current = phi_t
        S_current = deft_core.action(phi_current, R, Delta, t, N) * (N / G)

        # Do Monte Carlo sampling
        for k in range(
                                num_thermalization_steps + num_samples * num_steps_per_sample + 1):
            i = np.random.randint(0, G)
            eig_val = eig_vals[i]
            eig_vec = eig_vecs[:, i]
            step_size = np.random.normal(0, 1.0 / np.sqrt(eig_val))
            phi_new = phi_current + eig_vec * step_size
            S_new = deft_core.action(phi_new, R, Delta, t, N) * (N / G)
            if np.log(np.random.uniform(0, 1)) < (S_current - S_new):
                phi_current = phi_new
                S_current = S_new
            if (k > num_thermalization_steps) and (
                    k % num_steps_per_sample == 0):
                phi_samples[:, sample_index] = phi_current
                sample_index += 1

    # Return phi samples and phi weights
    return phi_samples, np.ones(num_samples)


# Sample probable densities using posterior probability
def posterior_sampling(points, R, Delta, N, G, num_pt_samples, fix_t_at_t_star):

    method, go_parallel = Laplace_approach, False

    phi_samples = np.zeros([G, num_pt_samples])
    phi_weights = np.zeros(num_pt_samples)
    sample_index = 0

    # Read in t, phi, log_E, and w_sample_mean from MAP curve points
    ts = sp.array([p.t for p in points])
    phis = sp.array([p.phi for p in points])
    log_Es = sp.array([p.log_E for p in points])
    w_sample_means = sp.array([p.sample_mean for p in points])

    # Generate a "histogram" of t according to their relative probability
    num_t = len(ts)
    if fix_t_at_t_star:
        hist_t = np.zeros(num_t)
        hist_t[log_Es.argmax()] = num_pt_samples
    else:
        log_Es = log_Es - log_Es.max()
        prob_t = sp.exp(log_Es)
        prob_t = prob_t / sp.sum(prob_t)
        num_indices = num_t
        sampled_indices = list(
            np.random.choice(num_indices, size=num_pt_samples, replace=True,
                             p=prob_t))
        hist_t = [sampled_indices.count(c) for c in range(num_indices)]

    # Traverse through t, and draw a number of phi samples for each t
    for i in range(num_t):
        num_samples = int(hist_t[i])
        if num_samples > 0:
            t = ts[i]
            phi_t = phis[i]
            phi_samples_at_t, phi_weights_at_t = \
                method(phi_t, R, Delta, t, N, num_samples, go_parallel,
                       pt_sampling=True)
            for k in range(num_samples):
                phi_samples[:, sample_index] = phi_samples_at_t[:, k]

                # JBK: I don't understand this
                phi_weights[sample_index] = phi_weights_at_t[k] / \
                                            w_sample_means[i]

                sample_index += 1

    # Convert phi samples to Q samples
    Q_samples = np.zeros([G, num_pt_samples])
    for k in range(num_pt_samples):
        Q_samples[:, k] = utils.field_to_prob(
            sp.array(phi_samples[:, k]).ravel())

    # Return Q samples along with their weights
    return Q_samples, phi_samples, phi_weights

