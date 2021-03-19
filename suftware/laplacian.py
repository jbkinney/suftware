# Generally, Python programs should be written with the assumption that all
# users are consenting adults, and thus are responsible for using things
# correctly themselves. (Silas Ray, StackOverflow)

import numpy as np

def bilateral_laplacian(alpha, npts):
    """
    Compute a numerical matrix representation of the discrete "bilateral
    laplacian" defined as u'*u product, where u is a matrix i.e. u*v gives
    the numerical first derivative of v (assuming that v is an vector etc.)
    of the order alpha.

    This implementation use forward differentiation finite differences method.

    Args:
        alpha (int): the order of the operator, should be greater than 0.
        npts (int): the number of gridpoints, should be reasonably great.

    Returns:
        the matrix numerical representation of the "bilateral laplacian".
    """
    u = np.eye(npts)
    for a in range(alpha):
        m = np.eye(npts - a) - np.eye(npts - a, npts - a, -1)
        u = m[1:,:] * u
    return u.T * u
