import scipy as sp
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import det, eigh, eigvalsh
import time

# python 3 imports
from suftware.src import utils
from suftware.src import supplements
from suftware.src import maxent

# Import error handling
from suftware.src.utils import ControlledError

# Put hard bounds on how big or small t can be. T_MIN especially seems to help convergence
T_MAX = 40
T_MIN = -40
PHI_MAX = utils.PHI_MAX
PHI_MIN = utils.PHI_MIN
MAX_DS = -1E-3
PHI_STD_REG = utils.PHI_STD_REG


class Results():
    pass


# Represents a point along the MAP curve
class MAP_curve_point:
    def __init__(self, t, phi, Q, log_E, log_Z_correction, sample_mean, sample_mean_std_dev, details=False):
        self.t = t
        self.phi = phi
        self.Q = Q
        self.log_E = log_E
        self.log_Z_correction = log_Z_correction
        self.sample_mean = sample_mean
        self.sample_mean_std_dev = sample_mean_std_dev
        # self.details = details


# Represents the MAP curve
class MAP_curve:
    def __init__(self):
        self.points = []
        self._is_sorted = False

    def add_point(self, t, phi, Q, log_E, log_Z_correction, sample_mean, sample_mean_std_dev, details=False):
        point = MAP_curve_point(t, phi, Q, log_E, log_Z_correction, sample_mean, sample_mean_std_dev, details)
        self.points.append(point)
        self._is_sorted = False

    def sort(self):
        self.points.sort(key=lambda x: x.t)
        self._is_sorted = True

    # Use this to get actual points along the MAP curve. This ensures that points are sorted
    def get_points(self):
        if not self._is_sorted:
            self.sort()
        return self.points

    def get_maxent_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[0]
        if not (p.t == -sp.Inf):
            raise ControlledError('/MAP_curve/ Not getting MaxEnt point: t = %f' % p.t)
        return p

    def get_histogram_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[-1]
        if not (p.t == sp.Inf):
            raise ControlledError('/MAP_curve/ Not getting histogram point: t = %f' % p.t)
        return p

    def get_log_evidence_ratios(self, finite=True):
        log_Es = sp.array([p.log_E for p in self.points])
        ts = sp.array([p.t for p in self.points])
        if finite:
            indices = (log_Es > -np.Inf) * (ts > -np.Inf) * (ts < np.Inf)
            return log_Es[indices], ts[indices]
        else:
            return log_Es, ts


#
# Convention: action, gradient, and hessian are G/N * the actual. This provides for more robust numerics
#
# Evaluate the action of a field given smoothness criteria
def action(phi, R, Delta, t, N, phi_in_kernel=False, regularized=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/action/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/action/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/action/ t is not real: t = %s' % t)
    # if not np.isfinite(t):
    #    raise ControlledError('/action/ t is not finite: t = %s' % t)
    # Make sure phi_in_kernel is valid
    if not isinstance(phi_in_kernel, bool):
        raise ControlledError('/action/ phi_in_kernel must be a boolean: phi_in_kernel = %s' % type(phi_in_kernel))
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/action/ regularized must be a boolean: regularized = %s' % type(regularized))

    G = 1. * len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    ones_col = sp.mat(sp.ones(int(G))).T

    if phi_in_kernel:
        S_mat = G * R_col.T * phi_col + G * ones_col.T * quasiQ_col
    else:
        S_mat = 0.5 * sp.exp(
            -t) * phi_col.T * Delta_sparse * phi_col + G * R_col.T * phi_col + G * ones_col.T * quasiQ_col

    if regularized:
        S_mat += 0.5 * (phi_col.T * phi_col) / (N * PHI_STD_REG ** 2)

    S = S_mat[0, 0]

    # Make sure S is valid
    if not np.isreal(S):
        raise ControlledError('/action/ S is not real at t = %s: S = %s' % (t, S))
    if not np.isfinite(S):
        raise ControlledError('/action/ S is not finite at t = %s: S = %s' % (t, S))

    return S


# Evaluate action gradient w.r.t. a field given smoothness criteria
def gradient(phi, R, Delta, t, N, regularized=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/gradient/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/gradient/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/gradient/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/gradient/ t is not finite: t = %s' % t)
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/gradient/ regularized must be a boolean: regularized = %s' % type(regularized))

    G = 1. * len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    grad_col = sp.exp(-t) * Delta_sparse * phi_col + G * R_col - G * quasiQ_col

    if regularized:
        grad_col += phi_col / (N * PHI_STD_REG ** 2)

    grad = sp.array(grad_col).ravel()

    # Make sure grad is valid
    if not all(np.isreal(grad)):
        raise ControlledError('/gradient/ grad is not real at t = %s: grad = %s' % (t, grad))
    if not all(np.isfinite(grad)):
        raise ControlledError('/gradient/ grad is not finite at t = %s: grad = %s' % (t, grad))

    return grad


# Evaluate action hessian w.r.t. a field given smoothness criteria. NOTE: returns sparse matrix, not dense matrix!
def hessian(phi, R, Delta, t, N, regularized=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/hessian/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/hessian/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/hessian/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/hessian/ t is not finite: t = %s' % t)
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/hessian/ regularized must be a boolean: regularized = %s' % type(regularized))

    G = 1. * len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    Delta_sparse = Delta.get_sparse_matrix()
    H = sp.exp(-t) * Delta_sparse + G * diags(quasiQ, 0)

    if regularized:
        H += diags(np.ones(int(G)), 0) / (N * PHI_STD_REG ** 2)

    # Make sure H is valid ?
    return H


# Compute the log of ptgd at maxent
def log_ptgd_at_maxent(phi_M, R, Delta, N, Z_eval, num_Z_samples):
    # Make sure phi_M is valid
    if not all(np.isreal(phi_M)):
        raise ControlledError('/log_ptgd_at_maxent/ phi_M is not real: phi_M = %s' % phi_M)
    if not all(np.isfinite(phi_M)):
        raise ControlledError('/log_ptgd_at_maxent/ phi_M is not finite: phi_M = %s' % phi_M)

    kernel_dim = Delta._kernel_dim
    M = utils.field_to_prob(phi_M)
    M_on_kernel = sp.zeros([kernel_dim, kernel_dim])
    kernel_basis = Delta._kernel_basis
    lambdas = Delta._eigenvalues
    for a in range(int(kernel_dim)):
        for b in range(int(kernel_dim)):
            psi_a = sp.ravel(kernel_basis[:, a])
            psi_b = sp.ravel(kernel_basis[:, b])
            M_on_kernel[a, b] = sp.sum(psi_a * psi_b * M)

    # Compute log Occam factor at infinity
    log_Occam_at_infty = - 0.5 * sp.log(det(M_on_kernel)) - 0.5 * sp.sum(sp.log(lambdas[kernel_dim:]))

    # Make sure log_Occam_at_infty is valid
    if not np.isreal(log_Occam_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_Occam_at_infty is not real: log_Occam_at_infty = %s' %
                              log_Occam_at_infty)
    if not np.isfinite(log_Occam_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_Occam_at_infty is not finite: log_Occam_at_infty = %s' %
                              log_Occam_at_infty)

    # Compute the log likelihood at infinity
    log_likelihood_at_infty = - N * sp.sum(phi_M * R) - N

    # Make sure log_likelihood_at_infty is valid
    if not np.isreal(log_likelihood_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_likelihood_at_infty is not real: log_likelihood_at_infty = %s' %
                              log_likelihood_at_infty)
    if not np.isfinite(log_likelihood_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_likelihood_at_infty is not finite: log_likelihood_at_infty = %s' %
                              log_likelihood_at_infty)

    # Compute the log posterior (not sure this is right)
    log_ptgd_at_maxent = log_likelihood_at_infty + log_Occam_at_infty

    # If requested, incorporate corrections to the partition function
    t = -np.inf
    num_samples = num_Z_samples
    if Z_eval == 'Lap':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            0.0, 1.0, 0.0
    if Z_eval == 'Lap+Imp':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=False)
    if Z_eval == 'Lap+Imp+P':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=True)
    if Z_eval == 'GLap':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=False, sampling=False)
    if Z_eval == 'GLap+P':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=True, sampling=False)
    if Z_eval == 'GLap+Sam':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=False, sampling=True)
    if Z_eval == 'GLap+Sam+P':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=True, sampling=True)
    if Z_eval == 'Lap+Fey':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.Feynman_diagrams(phi_M, R, Delta, t, N)

    # Make sure log_Z_correction is valid
    if not np.isreal(log_Z_correction):
        raise ControlledError('/log_ptgd_at_maxent/ log_Z_correction is not real: correction = %s' % correction)
    if not np.isfinite(log_Z_correction):
        raise ControlledError('/log_ptgd_at_maxent/ log_Z_correction is not finite: correction = %s' % correction)

    log_ptgd_at_maxent += log_Z_correction

    return log_ptgd_at_maxent, log_Z_correction, w_sample_mean, w_sample_mean_std


# Computes the log of ptgd at t
def log_ptgd(phi, R, Delta, t, N, Z_eval, num_Z_samples):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/log_ptgd/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/log_ptgd/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/log_ptgd/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/log_ptgd/ t is not finite: t = %s' % t)

    G = 1. * len(phi)
    alpha = 1. * Delta._alpha
    kernel_dim = 1. * Delta._kernel_dim
    H = hessian(phi, R, Delta, t, N)
    H_prime = H.todense() * sp.exp(t)

    S = action(phi, R, Delta, t, N)

    # First try computing log determinant straight away
    log_det = sp.log(det(H_prime))

    # If failed, try computing the sum of eigenvalues, forcing the eigenvalues to be real and non-negative
    if not (np.isreal(log_det) and np.isfinite(log_det)):
        lambdas = abs(eigvalsh(H_prime))
        log_det = sp.sum(sp.log(lambdas))

        # Make sure log_det is valid
    if not np.isreal(log_det):
        raise ControlledError('/log_ptgd/ log_det is not real at t = %s: log_det = %s' % (t, log_det))
    if not np.isfinite(log_det):
        raise ControlledError('/log_ptgd/ log_det is not finite at t = %s: log_det = %s' % (t, log_det))

    # Compute contribution from finite t
    log_ptgd = -(N / G) * S + 0.5 * kernel_dim * t - 0.5 * log_det

    # Make sure log_ptgd is valid
    if not np.isreal(log_ptgd):
        raise ControlledError('/log_ptgd/ log_ptgd is not real at t = %s: log_ptgd = %s' % (t, log_ptgd))
    if not np.isfinite(log_ptgd):
        raise ControlledError('/log_ptgd/ log_ptgd is not finite at t = %s: log_ptgd = %s' % (t, log_ptgd))

    # If requested, incorporate corrections to the partition function
    num_samples = num_Z_samples
    if Z_eval == 'Lap':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            0.0, 1.0, 0.0
    if Z_eval == 'Lap+Imp':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=False)
    if Z_eval == 'Lap+Imp+P':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=True)
    if Z_eval == 'GLap':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=False, sampling=False)
    if Z_eval == 'GLap+P':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=True, sampling=False)
    if Z_eval == 'GLap+Sam':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=False, sampling=True)
    if Z_eval == 'GLap+Sam+P':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=True, sampling=True)
    if Z_eval == 'Lap+Fey':
        log_Z_correction, w_sample_mean, w_sample_mean_std = \
            supplements.Feynman_diagrams(phi, R, Delta, t, N)

    # Make sure log_Z_correction is valid
    if not np.isreal(log_Z_correction):
        raise ControlledError('/log_ptgd/ log_Z_correction is not real at t = %s: correction = %s' % (t, correction))
    if not np.isfinite(log_Z_correction):
        raise ControlledError('/log_ptgd/ log_Z_correction is not finite at t = %s: correction = %s' % (t, correction))

    log_ptgd += log_Z_correction

    details = Results()
    details.S = S
    details.N = N
    details.G = G
    details.kernel_dim = kernel_dim
    details.t = t
    details.log_det = log_det
    details.phi = phi

    return log_ptgd, log_Z_correction, w_sample_mean, w_sample_mean_std


# Computes predictor step
def compute_predictor_step(phi, R, Delta, t, N, direction, resolution, DT_MAX):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/compute_predictor_step/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/compute_predictor_step/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/compute_predictor_step/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/compute_predictor_step/ t is not finite: t = %s' % t)
    # Make sure direction is valid
    if not ((direction == 1) or (direction == -1)):
        raise ControlledError('/compute_predictor_step/ direction must be just a sign: direction = %s' % direction)

    # Get current probability distribution
    Q = utils.field_to_prob(phi)
    G = 1. * len(Q)

    # Get hessian
    H = hessian(phi, R, Delta, t, N)

    # Compute rho, which indicates direction of step
    rho = G * spsolve(H, Q - R)

    # Make sure rho is valid
    if not all(np.isreal(rho)):
        raise ControlledError('/compute_predictor_step/ rho is not real at t = %s: rho = %s' % (t, rho))
    if not all(np.isfinite(rho)):
        raise ControlledError('/compute_predictor_step/ rho is not finite at t = %s: rho = %s' % (t, rho))

    denom = sp.sqrt(sp.sum(rho * Q * rho))

    # Make sure denom is valid
    if not np.isreal(denom):
        raise ControlledError('/compute_predictor_step/ denom is not real at t = %s: denom = %s' % (t, denom))
    if not np.isfinite(denom):
        raise ControlledError('/compute_predictor_step/ denom is not finite at t = %s: denom = %s' % (t, denom))
    if not (denom > 0):
        raise ControlledError('/compute_predictor_step/ denom is not positive at t = %s: denom = %s' % (t, denom))

    # Compute dt based on value of epsilon (the resolution)
    dt = direction * resolution / denom
    while abs(dt) > DT_MAX:
        dt /= 2.0

        # Return phi_new and new t_new. WARNING: IT IS NOT YET CLEAR THAT PHI_NEW ISN'T INSANE
    phi_new = phi + rho * dt
    t_new = t + dt

    # Make sure phi_new is valid
    if not all(np.isreal(phi_new)):
        raise ControlledError('/compute_predictor_step/ phi_new is not real at t_new = %s: phi_new = %s' % (t_new, phi_new))
    if not all(np.isfinite(phi_new)):
        raise ControlledError('/compute_predictor_step/ phi_new is not finite at t_new = %s: phi_new = %s' % (t_new, phi_new))
    # Make sure t_new is valid
    if not np.isreal(t_new):
        raise ControlledError('/compute_predictor_step/ t_new is not real: t_new = %s' % t_new)
    if not np.isfinite(t_new):
        raise ControlledError('/compute_predictor_step/ t_new is not finite: t_new = %s' % t_new)

    return phi_new, t_new


# Computes corrector step
def compute_corrector_step(phi, R, Delta, t, N, tollerance, report_num_steps=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/compute_corrector_step/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/compute_corrector_step/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/compute_corrector_step/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/compute_corrector_step/ t is not finite: t = %s' % t)
    # Make sure report_num_steps is valid
    if not isinstance(report_num_steps, bool):
        raise ControlledError('/compute_corrector_step/ report_num_steps must be a boolean: report_num_steps = %s' %
                              type(report_num_steps))

    # Evaluate the probability distribution
    Q = utils.field_to_prob(phi)

    # Evaluate action
    S = action(phi, R, Delta, t, N)

    # Perform corrector steps until phi converges
    num_corrector_steps = 0
    num_backtracks = 0
    while True:

        # Compute the gradient
        v = gradient(phi, R, Delta, t, N)

        # Compute the hessian
        H = hessian(phi, R, Delta, t, N)

        # Solve linear equation to get change in field
        dphi = -spsolve(H, v)

        # Make sure dphi is valid
        if not all(np.isreal(dphi)):
            raise ControlledError('/compute_corrector_step/ dphi is not real at t = %s: dphi = %s' % (t, dphi))
        if not all(np.isfinite(dphi)):
            raise ControlledError('/compute_corrector_step/ dphi is not finite at t = %s: dphi = %s' % (t, dphi))

        # Compute corresponding change in action
        dS = sp.sum(dphi * v)

        # If we're already very close to the max, then dS will be close to zero. In this case, we're done already
        if dS > MAX_DS:
            break

        # Reduce step size until in linear regime
        beta = 1.0
        while True:

            # Make sure beta is valid
            if beta < 1E-50:
                raise ControlledError('/compute_corrector_step/ phi is not converging at t = %s: beta = %s' % (t, beta))

            # Compute new phi
            phi_new = phi + beta * dphi

            # If new phi is insane, decrease beta
            if any(phi_new < PHI_MIN) or any(phi_new > PHI_MAX):
                num_backtracks += 1
                beta *= 0.5
                continue

            # Compute new action
            S_new = action(phi_new, R, Delta, t, N)

            # Check for linear regime
            if S_new - S <= 0.5 * beta * dS:
                break

            # If not in linear regime, backtrack value of beta
            else:
                num_backtracks += 1
                beta *= 0.5
                continue

        # Make sure phi_new is valid
        if not all(np.isreal(phi_new)):
            raise ControlledError('/compute_corrector_step/ phi_new is not real at t = %s: phi_new = %s' % (t, phi_new))
        if not all(np.isfinite(phi_new)):
            raise ControlledError('/compute_corrector_step/ phi_new is not finite at t = %s: phi_new = %s' % (t, phi_new))

        # Compute new Q
        Q_new = utils.field_to_prob(phi_new)

        # Break out of loop if Q_new is close enough to Q
        gd = utils.geo_dist(Q_new, Q)
        if gd < tollerance:
            break

        # Break out of loop with warning if S_new > S.
        # Should not happen, but not fatal if it does. Just means less precision
        # ACTUALLY, THIS SHOULD NEVER HAPPEN!
        elif S_new - S > 0:
            raise ControlledError('/compute_corrector_step/ S_new > S at t = %s: terminating corrector steps' % t)

        # Otherwise, continue with corrector step
        else:
            # New phi, Q, and S values have already been computed
            phi = phi_new
            Q = Q_new
            S = S_new
            num_corrector_steps += 1

    # After corrector loop has finished, return field
    if report_num_steps:
        return phi, num_corrector_steps, num_backtracks
    else:
        return phi


# The core algorithm of DEFT, used for both 1D and 2D density estimation
def compute_map_curve(N, R, Delta, Z_eval, num_Z_samples, t_start, DT_MAX, print_t, tollerance, resolution, max_log_evidence_ratio_drop):
    """ Traces the map curve in both directions

    Args:

        R (numpy.narray):
            The data histogram

        Delta (Smoothness_operator instance):
            Effectiely defines smoothness

        resolution (float):
            Specifies max distance between neighboring points on the
            MAP curve

    Returns:

        map_curve (list): A list of MAP_curve_points

    """

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    alpha = Delta._alpha
    kernel_basis = Delta.get_kernel_basis()
    kernel_dim = Delta.get_kernel_dim()

    # Initialize MAP curve
    map_curve = MAP_curve()

    #
    # First compute histogram stuff
    #

    # Get normalized histogram and corresponding field
    R = R / sum(R)
    phi_R = utils.prob_to_field(R)
    log_E_R = -np.Inf
    t_R = np.Inf
    log_Z_correction_R = 0.0
    w_sample_mean_R = 1.0
    w_sample_mean_std_R = 0.0
    map_curve.add_point(t_R, phi_R, R, log_E_R, log_Z_correction_R, w_sample_mean_R, w_sample_mean_std_R)

    #
    # Then compute maxent stuff
    #

    # Compute the maxent field and density
    phi_infty, success = maxent.compute_maxent_field(R, kernel_basis)

    # Convert maxent field to probability distribution
    M = utils.field_to_prob(phi_infty)

    # Compute the maxent log_ptgd. Important to keep this around to compute log_E at finite t
    log_ptgd_M, log_Z_correction_M, w_sample_mean_M, w_sample_mean_std_M = \
        log_ptgd_at_maxent(phi_infty, R, Delta, N, Z_eval, num_Z_samples)

    # This corresponds to a log_E of zero
    log_E_M = 0
    t_M = -sp.Inf
    map_curve.add_point(t_M, phi_infty, M, log_E_M, log_Z_correction_M, w_sample_mean_M, w_sample_mean_std_M)

    # Set maximum log evidence ratio so far encountered
    log_E_max = -np.Inf

    #
    # Now compute starting point
    #

    # Compute phi_start by executing a corrector step starting at maxent dist
    phi_start = compute_corrector_step(phi_infty, R, Delta, t_start, N, tollerance)

    # Convert starting field to probability distribution
    Q_start = utils.field_to_prob(phi_start)

    # Compute log ptgd
    log_ptgd_start, log_Z_correction_start, w_sample_mean_start, w_sample_mean_std_start = \
        log_ptgd(phi_start, R, Delta, t_start, N, Z_eval, num_Z_samples)

    # Compute corresponding evidence ratio
    log_E_start = log_ptgd_start - log_ptgd_M

    # Adjust max log evidence ratio
    log_E_max = log_E_start if (log_E_start > log_E_max) else log_E_max

    # Set start as first MAP curve point
    if print_t:
        print('t = %.2f' % t_start)
    map_curve.add_point(t_start, phi_start, Q_start, log_E_start, log_Z_correction_start, w_sample_mean_start, w_sample_mean_std_start)

    #
    # Finally trace along the MAP curve
    #

    # This is to indicate how iteration in t is terminated
    break_t_loop = [True, True]  # = [Q_M, Q_R]; True = thru geo_dist, False = thru log_E

    # Trace MAP curve in both directions
    for direction in [-1, +1]:

        # Start iteration from central point
        phi = phi_start
        t = t_start
        Q = Q_start
        log_E = log_E_start
        log_Z_correction = log_Z_correction_start
        w_sample_mean = w_sample_mean_start
        w_sample_mean_std_dev = w_sample_mean_std_start

        if direction == -1:
            Q_end = M
        else:
            Q_end = R

        log_ptgd0 = log_ptgd_start
        slope = np.sign(0)

        # Keep stepping in direction until reach the specified endpoint
        while True:

            # Test distance to endpoint
            if utils.geo_dist(Q_end, Q) <= resolution:
                if direction == -1:
                    pass
                    #print('Q_end = M: geo_dist (%.2f) <= resolution (%.2f)' % (utils.geo_dist(Q_end, Q), resolution))
                else:
                    pass
                    #print('Q_end = R: geo_dist (%.2f) <= resolution (%.2f)' % (utils.geo_dist(Q_end, Q), resolution))
                break

            # Take predictor step
            phi_pre, t_new = compute_predictor_step(phi, R, Delta, t, N, direction, resolution, DT_MAX)

            # If phi_pre is insane, start iterating from phi instead
            if any(phi_pre > PHI_MAX) or any(phi_pre < PHI_MIN):
                phi_pre = phi

            # Perform corrector steps to get new phi
            phi_new = compute_corrector_step(phi_pre, R, Delta, t_new, N, tollerance)

            # Compute new distribution
            Q_new = utils.field_to_prob(phi_new)

            # Compute log ptgd
            log_ptgd_new, log_Z_correction_new, w_sample_mean_new, w_sample_mean_std_new = \
                log_ptgd(phi_new, R, Delta, t_new, N, Z_eval, num_Z_samples)

            # Compute corresponding evidence ratio
            log_E_new = log_ptgd_new - log_ptgd_M

            # Take step
            t = t_new
            Q = Q_new
            phi = phi_new
            log_E = log_E_new
            log_Z_correction = log_Z_correction_new
            w_sample_mean = w_sample_mean_new
            w_sample_mean_std = w_sample_mean_std_new

            # Adjust max log evidence ratio
            log_E_max = log_E if (log_E > log_E_max) else log_E_max

            # Terminate if log_E is too small. But don't count the t=-inf endpoint when computing log_E_max
            if log_E_new < log_E_max - max_log_evidence_ratio_drop:
                if direction == -1:
                    #print('Q_end = M: log_E (%.2f) < log_E_max (%.2f) - max_log_evidence_ratio_drop (%.2f)' %
                    #      (log_E_new, log_E_max, max_log_evidence_ratio_drop))
                    break_t_loop[0] = False
                else:
                    #print('Q_end = R: log_E (%.2f) < log_E_max (%.2f) - max_log_evidence_ratio_drop (%.2f)' %
                    #      (log_E_new, log_E_max, max_log_evidence_ratio_drop))
                    break_t_loop[1] = False
                # Add new point to map curve
                if print_t:
                    print('t = %.2f' % t)
                map_curve.add_point(t, phi, Q, log_E, log_Z_correction, w_sample_mean, w_sample_mean_std)
                break

            slope_new = np.sign(log_ptgd_new - log_ptgd0)
            # Terminate if t is too negative or too positive
            if t < T_MIN:
                #print('Q_end = M: t (%.2f) < T_MIN (%.2f)' % (t, T_MIN))
                break_t_loop[0] = False
                break
            elif t > T_MAX:
                #print('Q_end = R: t (%.2f) > T_MAX (%.2f)' % (t, T_MAX))
                break_t_loop[1] = False
                break
            elif (direction == +1) and (t > 0) and (np.sign(slope_new * slope) < 0) and (log_ptgd_new > log_ptgd0):
                #print('Q_end = R: t (%.2f) > 0 and log_ptgd_new (%.2f) > log_ptgd (%.2f) wrongly' %
                #      (t, log_ptgd_new, log_ptgd0))
                break_t_loop[1] = False
                break
            elif (direction == +1) and (np.sign(slope_new * slope) < 0) and (log_ptgd_new > log_ptgd0 + max_log_evidence_ratio_drop):
                #print('Q_end = R: log_ptgd_new (%.2f) > log_ptgd (%.2f) + max_log_evidence_ratio_drop (%.2f) at t = %.2f' %
                #      (log_ptgd_new, log_ptgd0, max_log_evidence_ratio_drop, t))
                break_t_loop[1] = False
                break
            log_ptgd0 = log_ptgd_new
            slope = slope_new

            # Add new point to MAP curve
            if print_t:
                print('t = %.2f' % t)
            map_curve.add_point(t, phi, Q, log_E, log_Z_correction, w_sample_mean, w_sample_mean_std)

    # Sort points along the MAP curve
    map_curve.sort()
    map_curve.t_start = t_start
    map_curve.break_t_loop = break_t_loop

    # Return the MAP curve to the user
    return map_curve



#
# Compute the K coefficient (Kinney 2015 PRE, Eq. 12)
#

def _compute_K_coeff(res):

    # Compute the spectrum of Delta
    Delta = res.Delta.get_dense_matrix()
    alpha = int(-Delta[0, 1])
    lambdas, psis = eigh(Delta)  # Columns of psi are eigenvectors
    original_psis = sp.array(psis)

    R = res.R
    M = res.M
    N = res.N
    G = len(R)

    # Get normalized M and R, with unit grid spacing
    M = sp.array(M / sp.sum(M)).T
    R = sp.array(R / sp.sum(R)).T

    # Diagonalize first alpha psis with respect to diag_M
    # This does the trick
    diag_M_mat = sp.mat(sp.diag(M))
    psis_ker_mat = sp.mat(original_psis[:, :alpha])
    diag_M_ker = psis_ker_mat.T * diag_M_mat * psis_ker_mat
    omegas, psis_ker_coeffs = eigh(diag_M_ker)

    psis = original_psis.copy()
    psis[:, :alpha] = psis_ker_mat * psis_ker_coeffs

    # Now compute relevant coefficients
    # i: range(G)
    # j,k: range(alpha)
    v_is = sp.array([sp.sum((M - R) * psis[:, i]) for i in range(G)])
    z_iis = sp.array([sp.sum(M * psis[:, i] * psis[:, i]) for i in range(G)])
    z_ijs = sp.array(
        [[sp.sum(M * psis[:, i] * psis[:, j]) for j in range(alpha)] for i in
         range(G)])
    z_ijks = sp.array([[[sp.sum(M * psis[:, i] * psis[:, j] * psis[:, k]) for j
                         in range(alpha)] for k in range(alpha)] for i in
                       range(G)])

    K_pos_terms = sp.array(
        [(N * v_is[i] ** 2) / (2 * lambdas[i]) for i in range(alpha, G)])
    K_neg_terms = sp.array(
        [(-z_iis[i]) / (2 * lambdas[i]) for i in range(alpha, G)])
    K_ker1_terms = sp.array([sum(
        [z_ijs[i, j] ** 2 / (2 * lambdas[i] * omegas[j]) for j in range(alpha)])
                             for i in range(alpha, G)])
    K_ker2_terms = sp.array([sum(
        [v_is[i] * z_ijks[i, j, j] / (2 * lambdas[i] * omegas[j]) for j in
         range(alpha)]) for i in range(alpha, G)])
    K_ker3_terms = sp.array([sum([sum([-v_is[i] * z_ijs[i, j] * z_ijks[
        j, k, k] / (2 * lambdas[i] * omegas[k] * omegas[j]) for j in
                                       range(alpha)]) for k in range(alpha)])
                             for i in range(alpha, G)])

    # I THINK THIS IS RIGHT!!!
    K_coeff = K_pos_terms.sum() + K_neg_terms.sum() + K_ker1_terms.sum() + K_ker2_terms.sum() + K_ker3_terms.sum()

    # Return the K coefficient
    return K_coeff


#
# Core DEFT algorithm
#
def run(counts_array, Delta, Z_eval, num_Z_samples, t_start, DT_MAX, print_t,
        tollerance, resolution, num_pt_samples, fix_t_at_t_star,
        max_log_evidence_ratio_drop, compute_K_coeff, details=False):
    """
    The core algorithm of DEFT, used for both 1D and 2D density estmation.

    Args:
        counts_array (numpy.ndarray):
            A scipy array of counts. All counts must be nonnegative.

        Delta (Smoothness_operator instance):
            An operator providing the definition of 'smoothness' used by DEFT
    """

    # Make sure details is valid
    if not isinstance(details, bool):
        raise ControlledError('/deft_core._run/ details must be a boolean: details = %s' % type(details))

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    kernel_dim = Delta.get_kernel_dim()

    # Make sure counts_array is valid
    if not (len(counts_array) == G):
        raise ControlledError('/deft_core._run/ counts_array must have length %d: len(counts_array) = %d' %
                              (G, len(counts_array)))
    if not all(counts_array >= 0):
        raise ControlledError('/deft_core._run/ counts_array is not non-negative: counts_array = %s' % counts_array)
    if not (sum(counts_array > 0) > kernel_dim):
        raise ControlledError('/deft_core._run/ Only %d elements of counts_array contain data, less than kernel dimension %d' %
                              (sum(counts_array > 0), kernel_dim))

    # Get number of data points and normalize histogram
    N = sum(counts_array)
    R = 1.0 * counts_array / N

    #
    # Compute the MAP curve
    #

    start_time = time.time()
    map_curve = compute_map_curve(N, R, Delta, Z_eval, num_Z_samples, t_start, DT_MAX, print_t, tollerance, resolution,max_log_evidence_ratio_drop)
    end_time = time.time()
    map_curve_compute_time = end_time - start_time
    if print_t:
        print('MAP curve computation took %.2f sec' % map_curve_compute_time)

    # Identify the optimal density estimate
    points = map_curve.points
    log_Es = sp.array([p.log_E for p in points])
    log_E_max = log_Es.max()
    ibest = log_Es.argmax()
    star = points[ibest]
    Q_star = np.copy(star.Q)
    t_star = star.t
    phi_star = np.copy(star.phi)
    map_curve.i_star = ibest

    #
    # Do posterior sampling
    #

    if not (num_pt_samples == 0):
        Q_samples, phi_samples, phi_weights = \
            supplements.posterior_sampling(points, R, Delta, N, G,
                                           num_pt_samples, fix_t_at_t_star)
            


    #
    # Package results
    #

    # Create a container
    results = Results()

    # Fill in info that's guaranteed to be there
    results.Delta = Delta
    results.phi_star = phi_star
    results.Q_star = Q_star
    results.R = R
    results.map_curve = map_curve
    results.map_curve_compute_time = map_curve_compute_time
    results.G = G
    results.N = N
    results.t_star = t_star
    results.i_star = ibest
    results.counts = counts_array
    results.tollerance = tollerance
    results.resolution = resolution
    results.points = points

    # Get maxent point
    maxent_point = results.map_curve.get_maxent_point()
    results.M = maxent_point.Q / np.sum(maxent_point.Q)

    # Compute K coefficient if requested
    if compute_K_coeff:
        results.K_coeff = _compute_K_coeff(results)
    else:
        results.K_coeff = None


    # Include posterior sampling info if any sampling was performed
    if not (num_pt_samples == 0):
        results.num_pt_samples = num_pt_samples
        results.Q_samples = Q_samples
        results.phi_samples = phi_samples
        results.phi_weights = phi_weights

    # Return density estimate along with histogram on which it is based
    return results
