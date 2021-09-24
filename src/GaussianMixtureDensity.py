import numpy as np
import scipy.stats as ss
from suftware.src.utils import grid_info_from_bbox_and_G
from suftware.src.Density import Density
from suftware.src.utils import check, handle_errors

class GaussianMixtureDensity(Density):
    """
    Creates a Gaussian mixture density; derived from Density

    Parameters
    ----------

    bounding_box: ([float, float])
        Box in which to create density.

    weights: (np.ndarray)
        Relative mass of each Gaussian component

    mus: (np.ndarray)
        Means for each Gaussian component

    sigmas: (np.ndarray)
        Standard deviations for each Gaussian component.

    name: (str)
        A user-defined name for the distribution, if any.

    num_gridpoints: (int >= 2)
        Number of grid points on which to sample. Defaults to number of
        grid points used to create the density.

    """
    @handle_errors
    def __init__(self, bounding_box, weights, mus, sigmas, name=None,
                 num_gridpoints=10000):

        # Cast input variables
        weights = np.array(weights).ravel()
        mus = np.array(mus).ravel()
        sigmas = np.array(sigmas).ravel()
        num_gridpoints = int(num_gridpoints)

        # Make sure all input arrays are the same length
        num_clusters = len(weights)
        check(len(mus) == num_clusters,
              'len(mus) = %d does not match len(weights)=%d' %
              (len(mus), num_clusters))
        check(len(sigmas) == num_clusters,
              'len(sigmas) = %d does not match len(weights)=%d' %
              (len(sigmas), num_clusters))

        # Make sure values are valid
        check(all(np.isfinite(mus)), 'All mus must be finite; some arent.')
        check(all(weights >= 0), 'All weights must be nonnegative; some arent.')
        check(any(weights > 0), 'Some weights must be positive; none are.')
        check(all(sigmas > 0), 'All sigmas must be positive; some arent')

        # Make sure num_gridpoints is large enough
        check(num_gridpoints >= 2,
              'num_gridpoints = %d; must be >= 2.' % num_gridpoints)

        # Compute grid
        h, grid, bin_edges = grid_info_from_bbox_and_G(bbox=bounding_box,
                                                       G=num_gridpoints)

        # Compute density values
        values = np.zeros(num_gridpoints)
        for weight, mu, sigma in zip(weights, mus, sigmas):
            values += weight*ss.norm.pdf(x=grid, loc=mu, scale=sigma)
        values /= np.sum(h*values)

        # Call density constructor
        Density.__init__(self,
                         grid=grid,
                         values=values,
                         interpolation_method='cubic',
                         min_value=1E-20)

        # Record Gaussian-mixture-specific parameters
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights
        self.name = name
