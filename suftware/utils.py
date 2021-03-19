import sys
import numbers
from functools import wraps

import numpy as np
from numpy.polynomial.legendre import legval, legval2d

import scipy as sp

# A very small floating point number, used to prevent taking logs of 0
TINY_FLOAT64 = sp.finfo(sp.float64).tiny
TINY_FLOAT32 = sp.finfo(sp.float32).tiny
PHI_MIN = -500 
PHI_MAX = 500
PHI_STD_REG = 100.0
LISTLIKE = (list, np.ndarray, np.matrix, range)

# This is useful for testing whether something is a number
NUMBER = (int, float, int)

# This is useful for testing whether something is an array
ARRAY = (np.ndarray, list)

# Dummy class
class Dummy():
    def __init__(self):
        pass

# Define error handling
class ControlledError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


# Evaluate geodesic distance 
def geo_dist(P,Q):

    # Make sure P is valid
    if not all(np.isreal(P)):
        raise ControlledError('/geo_dist/ P is not real: P = %s' % P)
    if not all(np.isfinite(P)):
        raise ControlledError('/geo_dist/ P is not finite: P = %s' % P)
    if not all(P >= 0):
        raise ControlledError('/geo_dist/ P is not non-negative: P = %s' % P)
    if not any(P > 0):
        raise ControlledError('/geo_dist/ P is vanishing: P = %s' % P)
    # Make sure Q is valid
    if not all(np.isreal(Q)):
        raise ControlledError('/geo_dist/ Q is not real: Q = %s' % Q)
    if not all(np.isfinite(Q)):
        raise ControlledError('/geo_dist/ Q is not finite: Q = %s' % Q)
    if not all(Q >= 0):
        raise ControlledError('/geo_dist/ Q is not non-negative: Q = %s' % Q)
    if not any(Q > 0):
        raise ControlledError('/geo_dist/ Q is vanishing: Q = %s' % Q)

    # Enforce proper normalization
    P_prob = P/sp.sum(P) 
    Q_prob = Q/sp.sum(Q) 

    # Return geo-distance. Arc-cosine can behave badly if argument is too close to one, so prepare for this
    try:
        dist = 2*sp.arccos(sp.sum(sp.sqrt(P_prob*Q_prob)))
        if not np.isreal(dist):
            raise ControlledError('/geo_dist/ dist is not real: dist = %s' % dist)
        if not (dist >= 0):
            raise ControlledError('/geo_dist/ dist is not >= 0: dist = %s' % dist)
    except:
        if sp.sum(sp.sqrt(P_prob*Q_prob)) > 1 - TINY_FLOAT32:
            dist = 0
        else:
            raise ControlledError('/geo_dist/ dist cannot be computed correctly!')

    # Return geo-distance
    return dist


# Convert field to non-normalized quasi probability distribution
def field_to_quasiprob(raw_phi):
    phi = np.copy(raw_phi) 
    G = len(phi)

    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/field_to_quasiprob/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/field_to_quasiprob/ phi is not finite: phi = %s' % phi)
    
    if any(phi < PHI_MIN):
        phi[phi < PHI_MIN] = PHI_MIN      

    # Compute quasiQ
    quasiQ = sp.exp(-phi)/(1.*G)

    # Make sure quasiQ is valid
    if not all(np.isreal(quasiQ)):
        raise ControlledError('/field_to_quasiprob/ quasiQ is not real: quasiQ = %s' % quasiQ)
    if not all(np.isfinite(quasiQ)):
        raise ControlledError('/field_to_quasiprob/ quasiQ is not finite: quasiQ = %s' % quasiQ)
    if not all(quasiQ >= 0):
        raise ControlledError('/field_to_quasiprob/ quasiQ is not non-negative: quasiQ = %s' % quasiQ)

    # Return quasi probability distribution
    return quasiQ


# Convert field to normalized probability distribution
def field_to_prob(raw_phi): 
    phi = np.copy(raw_phi) 
    G = len(phi)

    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/field_to_prob/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/field_to_prob/ phi is not finite: phi = %s' % phi)

    # Re-level phi. NOTE: CHANGES PHI!
    phi -= min(phi)

    # Compute Q
    denom = sp.sum(sp.exp(-phi))
    Q = sp.exp(-phi)/denom

    # Make sure Q is valid
    if not all(np.isreal(Q)):
        raise ControlledError('/field_to_prob/ Q is not real: Q = %s' % Q)
    if not all(np.isfinite(Q)):
        raise ControlledError('/field_to_prob/ Q is not finite: Q = %s' % Q)
    if not all(Q >= 0):
        raise ControlledError('/field_to_prob/ Q is not non-negative: Q = %s' % Q)

    # Return probability
    return Q


# Convert probability distribution to field
def prob_to_field(Q):
    G = len(Q)

    # Make sure Q is valid
    if not all(np.isreal(Q)):
        raise ControlledError('/prob_to_field/ Q is not real: Q = %s' % Q)
    if not all(np.isfinite(Q)):
        raise ControlledError('/prob_to_field/ Q is not finite: Q = %s' % Q)
    if not all(Q >= 0):
        raise ControlledError('/prob_to_field/ Q is not non-negative: Q = %s' % Q)
    
    phi = -sp.log(G*Q + TINY_FLOAT64)

    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/prob_to_field/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/prob_to_field/ phi is not finite: phi = %s' % phi)

    # Return field
    return phi


def grid_info_from_bin_centers_1d(bin_centers):
    bin_centers = np.array(bin_centers)
    h = bin_centers[1]-bin_centers[0]
    bbox = [bin_centers[0]-h/2., bin_centers[-1]+h/2.]
    G = len(bin_centers)
    bin_edges = np.zeros(G+1)
    bin_edges[0] = bbox[0]
    bin_edges[-1] = bbox[1]
    bin_edges[1:-1] = bin_centers[:-1]+h/2.
    return bbox, h, bin_edges


def grid_info_from_bin_edges_1d(bin_edges):
    bin_edges = np.array(bin_edges)
    h = bin_edges[1]-bin_edges[1]
    bbox = [bin_edges[0], bin_edges[-1]]
    bin_centers = bin_edges[:-1]+h/2.
    return bbox, h, bin_centers


def grid_info_from_bbox_and_G(bbox, G):
    bin_edges = np.linspace(bbox[0], bbox[1], num=G+1, endpoint=True)
    h = bin_edges[1]-bin_edges[0]
    bin_centers = bin_edges[:-1]+h/2.

    return h, bin_centers, bin_edges


# Make a 1d histogram. Bounding box is optional
def histogram_counts_1d(data, G, bbox, normalized=False):

    # Make sure normalized is valid
    if not isinstance(normalized, bool):
        raise ControlledError('/histogram_counts_1d/ normalized must be a boolean: normalized = %s' % type(normalized))

    # data_spread = max(data) - min(data)
    #
    # # Set lower bound automatically if called for
    # if bbox[0] == -np.Inf:
    #     bbox[0] = min(data) - data_spread*0.2
    #
    # # Set upper bound automatically if called for
    # if bbox[1] == np.Inf:
    #     bbox[1] = max(data) + data_spread*0.2

    # Crop data to bounding box
    indices = (data >= bbox[0]) & (data < bbox[1])
    cropped_data = data[0]

    # Get grid info from bbox and G
    h, bin_centers, bin_edges = grid_info_from_bbox_and_G(bbox, G)

    # Make sure h is valid
    if not (h > 0):
        raise ControlledError('/histogram_counts_1d/ h must be > 0: h = %s' % h)
    # Make sure bin_centers is valid
    if not (len(bin_centers) == G):
        raise ControlledError('/histogram_counts_1d/ bin_centers must have length %d: len(bin_centers) = %d' %
                              (G,len(bin_centers)))
    # Make sure bin_edges is valid
    if not (len(bin_edges) == G+1):
        raise ControlledError('/histogram_counts_1d/ bin_edges must have length %d: len(bin_edges) = %d' %
                              (G+1,len(bin_edges)))

    # Get counts in each bin
    counts, _ = np.histogram(data, bins=bin_edges, density=False)

    # Make sure counts is valid
    if not (len(counts) == G):
        raise ControlledError('/histogram_counts_1d/ counts must have length %d: len(counts) = %d' % (G, len(counts)))
    if not all(counts >= 0):
        raise ControlledError('/histogram_counts_1d/ counts is not non-negative: counts = %s' % counts)
    
    if normalized:
        hist = 1.*counts/np.sum(h*counts)
    else:
        hist = counts

    # Return the number of counts and the bin centers
    return hist, bin_centers


# Make a 2d histogram
def histogram_2d(data, box, num_bins=[10,10], normalized=False):
    data_x = data[0]
    data_y = data[1]

    hx, xs, x_edges = \
        grid_info_from_bbox_and_G(box[0], num_bins[0])
    hy, ys, y_edges = \
        grid_info_from_bbox_and_G(box[1], num_bins[1])

    hist, xedges, yedges = np.histogram2d(data_x, data_y, 
        bins=[x_edges, y_edges], normed=normalized)

    return hist, xs, ys


# Returns the left edges of a binning given the centers
def left_edges_from_centers(centers):
    h = centers[1]-centers[0]
    return centers - h/2.


# Returns the domain of a binning given the centers
def bounding_box_from_centers(centers):
    h = centers[1]-centers[0]
    xmin = centers[0]-h/2.
    xmax = centers[-1]+h/2.
    return sp.array([xmin,xmax])


# Defines a dot product with my normalization
def dot(v1,v2,h=1.0):
    v1r = v1.ravel()
    v2r = v2.ravel()
    G = len(v1)
    if not (len(v2) == G):
        raise ControlledError('/dot/ vectors are not of the same length: len(v1) = %d, len(v2) = %d' % (len(v1r), len(v2r)))
    return sp.sum(v1r*v2r*h/(1.*G))


# Computes a norm with my normalization
def norm(v,h=1.0):
    v_cc = np.conj(v)
    return sp.sqrt(dot(v,v_cc,h))


# Normalizes vectors (stored as columns of a 2D numpy array)
def normalize(vectors, grid_spacing=1.0):
    """ Normalizes vectors stored as columns of a 2D numpy array """
    G = vectors.shape[0] # length of each vector
    K = vectors.shape[1] # number of vectors

    # Set volume element h. This takes a little consideration
    if isinstance(grid_spacing,NUMBER):
        h = grid_spacing
    elif isinstance(grid_spacing,ARRAY):
        grid_spacing = sp.array(grid_spacing)
        h = sp.prod(grid_spacing)
    else:
        raise ControlledError('/normalize/ Cannot recognize h: h = %s' % h)
    
    if not (h > 0):
        raise ControlledError('/normalize/ h is not positive: h = %s' % h)

    norm_vectors = sp.zeros([G,K])
    for i in range(K):
        # Extract v from list of vectors
        v = vectors[:,i]
        # Flip sign of v so that last element is non-negative
        if (v[-1] < 0):
            v = -v
        # Normalize v and save in norm_vectors
        norm_vectors[:,i] = v/norm(v)

    # Return array with normalized vectors along the columns
    return norm_vectors


# Construct an orthonormal basis of order alpha from 1d legendre polynomials
def legendre_basis_1d(G, alpha, grid_spacing):

    # Create grid of centred x-values ranging from -1 to 1
    x_grid = (sp.arange(G) - (G-1)/2.)/(G/2.)

    # First create an orthogonal (not necessarily normalized) basis
    raw_basis = sp.zeros([G,alpha])
    for i in range(alpha):
        c = sp.zeros(alpha)
        c[i] = 1.0
        raw_basis[:,i] = legval(x_grid,c)

    # Normalize basis
    basis = normalize(raw_basis, grid_spacing)

    # Return normalized basis
    return basis


# Construct an orthonormal basis of order alpha from 2d legendre polynomials
def legendre_basis_2d(Gx, Gy, alpha, grid_spacing=[1.0,1.0]):

    # Compute x-coords and y-coords, each ranging from -1 to 1
    x_grid = (sp.arange(Gx) - (Gx-1)/2.)/(Gx/2.)
    y_grid = (sp.arange(Gy) - (Gy-1)/2.)/(Gy/2.)

    # Create meshgrid of these
    xs, ys = np.meshgrid(x_grid,y_grid)
    basis_dim = alpha*(alpha+1)/2
    G = Gx*Gy
    raw_basis = sp.zeros([G,basis_dim])
    k = 0
    for a in range(alpha):
        for b in range(alpha):
            if a+b < alpha:
                c = sp.zeros([alpha,alpha])
                c[a,b] = 1
                raw_basis[:,k] = \
                    legval2d(xs,ys,c).T.reshape([G])
                k += 1

    # Normalize this basis using my convension
    basis = normalize(raw_basis, grid_spacing)
    
    # Return normalized basis
    return basis

def clean_numerical_input(x):
    """
    Returns a 1D np.array containing the numerical values in x or the
    value of x itself. Also returns well as a flag indicating whether x was
    passed as a single number or a list-like array.

    parameters
    ----------

    x: (number or list-like collection of numbers)
        The locations in the data domain at which to evaluate sampled
        density.

    returns
    -------

    x_arr: (1D np.array)
        Array containing numerical values of x.

    is_number: (bool)
        Flag indicating whether x was passed as a single number.

    """

    # If x is a number, record this fact and transform to np.array
    is_number = False
    if isinstance(x, numbers.Real):

        # Record that x is a single number
        is_number = True

        # Cast as 1D np.array of floats
        x = np.array([x]).astype(float)

    # Otherwise, if list-like, cast as 1D np.ndarray
    elif isinstance(x, LISTLIKE):

        # Cast as np.array
        x = np.array(x).ravel()

        # Check if x has any content
        check(len(x) > 0, 'x is empty.')

        # Check that entries are numbers
        check(all([isinstance(n, numbers.Real) for n in x]),
              'not all entries in x are real numbers')

        # Cast as 1D np.array of floats
        x = x.astype(float)

    # Otherwise, raise error
    else:
        raise ControlledError(
            'x is not a number or list-like, i.e., one of %s.'
            % str(LISTLIKE))

    # Make sure elements of x are all finite
    check(all(np.isfinite(x)),
          'Not all elements of x are finite.')

    return x, is_number


def check(condition, message):
    '''
    Checks a condition; raises a ControlledError with message if condition fails.
    :param condition:
    :param message:
    :return: None
    '''
    if not condition:
        raise ControlledError(message)


def enable_graphics(backend='TkAgg'):
    """
    Enable graphical output by suftware.

    This function should be _run before any calls are made to DensityEstimator.plot().
    This is not always necessary, since DensityEstimator.plot() itself will call this
    function if necessary. However, when plotting inline using the iPython
    notebook, this function must be called before the magic function
    ``%matplotlib inline``, e.g.::

        import suftware as sw
        sw.enable_graphics()
        %matplotlib inline

    If this function is never called, suftware can be _run without importing
    matplotlib. This can be useful, for instance, when distributing jobs
    across the nodes of a high performance computing cluster.


    parameters
    ----------

        backend: (str)
            Graphical backend to be passed to matplotlib.use().
            See the `matplotlib documentation <https://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`_
            for more information on graphical backends.


    returns
    -------

        None.

    """
    try:
        global mpl
        import matplotlib as mpl
        mpl.use(backend)
        global plt
        import matplotlib.pyplot as plt
    except:
        raise ControlledError('Could not import matplotlib.')


def handle_errors(func):
    """
    Decorator function to handle SUFTware errors
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):

        # Get should_fail debug flag
        should_fail = kwargs.pop('should_fail', None)

        try:

            # Execute function
            result = func(*args, **kwargs)
            error = False

            # If function didn't raise error, process results
            if should_fail is True:
                print('MISTAKE: Succeeded but should have failed.')
                mistake = True

            elif should_fail is False:
                print('Success, as expected.')
                mistake = False

            elif should_fail is None:
                mistake = False

            else:
                print('FATAL: should_fail = %s is not bool or None' %
                      should_fail)
                sys.exit(1)

        except ControlledError as e:
            error = True

            if should_fail is True:
                print('Error, as expected: ', e)
                mistake = False

            elif should_fail is False:
                print('MISTAKE: Failed but should have succeeded: ', e)
                mistake = True

            # Otherwise, just print an error and don't return anything
            else:
                print('Error: ', e)

        # If not in debug mode
        if should_fail is None:

            # If error, exit
            if error:
                sys.exit(1)

            # Otherwise, just return normal result
            else:
                return result

        # Otherwise, if in debug mode
        else:

            # If this is a constructor, set 'mistake' attribute of self
            if func.__name__ == '__init__':
                assert len(args) > 0
                args[0].mistake = mistake
                return None

            # Otherwise, create dummy object with mistake attribute
            else:
                obj = Dummy()
                obj.mistake = mistake
                return obj

    return wrapped_func
    