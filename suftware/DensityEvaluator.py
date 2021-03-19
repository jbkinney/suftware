"""
The definition of InterpolatedPDF factory class.
"""
#@todo: CHANGE FILE NAME TO interpolatedpdf.py OR place the content with other
#       fragments of the program code.

import numpy as np
import scipy as sp

class InterpolatedPDF:
    """
    The interpolation of the probabilility density function (PDF) is fabricated
    from given data by creating an InterpolatedPDF object. The object has got
    redefinied operator __call__(), i.e. it is function-like::

        r = 0.5  # an value of a random variable 
        p = InterpolatedPDF(x, y)
        probability = p(r)  # probability density value for given r
    """

    #? zmiana kolejności parametrów - było y,x ma być x,y
    #?
    def __init__(self, x, y, low=None, high=None, method='cubic'):
        """
        Args:
            x: abscissae, must be monotonic, i.e. x[n] < x[n + 1] for each n
            y: ordinates
            low (float): low limit for abscissae values
            high (float): high limit for abscissae values
            method: the method to be used by SciPy interpolate.interp1d,
                may be 'linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                'quadratic', 'cubic', 'previous' or 'next' (see also SciPy 1.6.0
                Reference Guide); default is 'cubic' and probably it is a safe
                choice.
        """

        self.__low = low
        self.__high = high

        # Compute normalization constant
        #
        step = x[1] - x[0]
        self.__z_normalization = step * np.sum(np.exp(-self._y))

        # Create an auxiliary interpolating function.
        #
        self.__aux_fun = sp.interpolate.interp1d(x, y, method, 
            bounds_error=False, fill_value='extrapolate', assume_sorted=True)

    def __f(self, x):
        """
        Compute distribution function value for scalar abscissa value.

        Args:
            x (float): an value
        """
        if self.__low <= x <= self.__high:
            return np.exp(-self.__aux_func(x)) / self.__z_normalization
        else:
            return 0.0

    def __call__(self, x):
        """
        Evaluates the probability density at specified positions.

        Args:
            x (np.array): locations at which to evaluate the density.

        Returns:
            np.array: values of the density at the specified positions.
                Values at positions outside the bounding box are evaluated to
                zero.
        """
        v = np.vectorize(self.__f)
        return v(x)
