import numpy as np
import scipy.stats as ss
from suftware.src.utils import grid_info_from_bbox_and_G
from suftware.src.Density import Density
from suftware.src.utils import check, handle_errors

class ScipyStatsDensity(Density):
    """
    Creates a SUFTware density from a built-in scipy.stats density

    Parameters
    ----------

    ss_obj:
        A scipy.stats density object, such as scipy.stats.beta or
        scipy.stats.gamma(a=3). Can be either frozen or not frozen.

    name: (str)
        A user-defined name for the distribution, if any.

    bounding_box: ([float, float])
        Box in which to create density. If None, bounding_box is automatically
        chosen to contain 99% of the distribution's mass.

    num_gridpoints: (int >= 2)
        Number of grid points on which to sample. Defaults to number of
        grid points used to create the density.

    **ss_obj_params:
        Keyword parameters, if any, taken by ss_obj.

    """

    @handle_errors
    def __init__(self, ss_obj, name=None, bounding_box=None, num_gridpoints=10000,
                 **ss_obj_params):

        # If bounding box is not specified, created one containing central
        # 99% of data
        if bounding_box is None:
            xmin = ss_obj.ppf(0.005, **ss_obj_params)
            xmax = ss_obj.ppf(0.995, **ss_obj_params)
            bounding_box = [xmin, xmax]

        # Compute grid
        h, grid, bin_edges = grid_info_from_bbox_and_G(bbox=bounding_box,
                                                       G=num_gridpoints)

        # Compute values on grid
        values = ss_obj.pdf(grid, **ss_obj_params)
        values /= sum(h*values)

        # Call density constructor
        Density.__init__(self,
                         grid=grid,
                         values=values,
                         interpolation_method='cubic',
                         min_value=1E-20)

        # Save ss_obj and parameters
        self.ss_obj = ss_obj
        self.ss_obj_params = ss_obj_params
        self.name = name
