#!/usr/local/bin/python -W ignore
import scipy as sp
import numpy as np
import sys
import time
import pdb
import numbers
import pandas as pd
import time

SMALL_NUM = 1E-6
MAX_NUM_GRID_POINTS = 1000
DEFAULT_NUM_GRID_POINTS = 100
MAX_NUM_POSTERIOR_SAMPLES = 1000
MAX_NUM_SAMPLES_FOR_Z = 1000000

# Import deft-related code
from suftware.src import deft_core
from suftware.src import laplacian
from suftware.src.utils import ControlledError, enable_graphics, check, handle_errors,\
    clean_numerical_input, LISTLIKE
from suftware.src.Density import Density

class DensityEstimator:
    """Estimates a 1D probability density from sampled data.

    parameters
    ----------
    data: (set, list, or np.array of numbers)
        An array of data from which the probability density will be estimated.
        Infinite or NaN values will be discarded.

    grid: (1D np.array)
        An array of evenly spaced grid points on which the probability density
        will be estimated. Default value is ``None``, in which case the grid is
        set automatically.

    grid_spacing: (float > 0)
        The distance at which to space neighboring grid points. Default value
        is ``None``, in which case this spacing is set automatically.

    num_grid_points: (int)
        The number of grid points to draw within the data domain. Restricted
        to ``2*alpha <= num_grid_points <= 1000``. Default value is ``None``, in
        which case the number of grid points is chosen automatically.

    bounding_box: ([float, float])
        The boundaries of the data domain, within which the probability density
        will be estimated. Default value is ``None``, in which case the
        bounding box is set automatically to encompass all of the data.

    alpha: (int)
        The order of derivative constrained in the definition of smoothness.
        Restricted to ``1 <= alpha <= 4``. Default value is 3.

    periodic: (bool)
        Whether or not to impose periodic boundary conditions on the estimated
        probability density. Default False, in which case no boundary
        conditions are imposed.

    num_posterior_samples: (int >= 0)
        Number of samples to draw from the Bayesian posterior. Restricted to
        0 <= num_posterior_samples <= MAX_NUM_POSTERIOR_SAMPLES.

    compute_K_coeff: (bool)
        Whether to compute the K coefficient (Kinney, 2015, PRE, Eq. 29),
        the sign of which tests the validity of the MaxEnt hypothesis on the
        data provided.

    max_t_step: (float > 0)
        Upper bound on the amount by which the parameter ``t``
        in the DEFT algorithm is incremented when tracing the MAP curve.
        Default value is 1.0.

    tollerance: (float > 0)
        Sets the convergence criterion for the corrector algorithm used in
        tracing the MAP curve.

    resolution: (float > 0)
        The maximum geodesic distance allowed for neighboring points
        on the MAP curve.

    sample_only_at_l_star: (boolean)
        Specifies whether to let l vary when sampling from the Bayesian
        posterior.

    max_log_evidence_ratio_drop: (float > 0)
        If set, MAP curve tracing will terminate prematurely when
        max_log_evidence - current_log_evidence >  max_log_evidence_ratio_drop.

    evaluation_method_for_Z: (string)
        Method of evaluation of partition function Z. Possible values:
        'Lap'      : Laplace approximation (default).
        'Lap+Imp'  : Laplace approximation + importance sampling.
        'Lap+Fey'  : Laplace approximation + Feynman diagrams.

    num_samples_for_Z: (int >= 0)
        Number of posterior samples to use when evaluating the paritation
        function Z. Only has an affect when
        ``evaluation_method_for_Z = 'Lap+Imp'``.

    seed: (int)
        Seed provided to the random number generator before density estimation
        commences. For development purposes only.

    print_t: (bool)
        Whether to print the values of ``t`` while tracing the MAP curve.
        For development purposes only.

    attributes
    ----------
    grid:
        The grid points at which the probability density was be estimated.
        (1D np.array)

    grid_spacing:
        The distance between neighboring grid points.
        (float > 0)

    num_grid_points:
        The number of grid points used.
        (int)

    bounding_box:
        The boundaries of the data domain within which the probability density
        was be estimated. ([float, float])

    histogram:
        A histogram of the data using ``grid`` for the centers of each bin.
        (1D np.array)

    values:
        The values of the optimal (i.e., MAP) density at each grid point.
        (1D np.array)

    sample_values:
        The values of the posterior sampled densities at each grid point.
        The first index specifies grid points, the second posterior samples.
        (2D np.array)

    sample_weights:
        The importance weights corresponding to each posterior sample.
        (1D np.array)

    K_coeff:
        The value of the K coefficient (Kinney, 2015, Eq. 29). (float)

    ells:
        The smoothness length scales at which the MAP curve was computed.
        (np.array)

    log_Es:
        The log evidence ratio values (Kinney, 2015, Eq. 27) at each length
        scale along the MAP curve. (np.array)

    max_log_E:
        The log evidence ratio at the optimal length scale. (float)

    runtime:
        The amount of time (in seconds) taken to execute.

    """

    @handle_errors
    def __init__(self,
                 data,
                 grid=None,
                 grid_spacing=None,
                 num_grid_points=None,
                 bounding_box=None,
                 alpha=3,
                 periodic=False,
                 num_posterior_samples=100,
                 compute_K_coeff=True,
                 t_start=None,
                 max_t_step=1.0,
                 tolerance=1E-6,
                 resolution=0.1,
                 sample_only_at_l_star=False,
                 max_log_evidence_ratio_drop=20,
                 evaluation_method_for_Z='Lap',
                 num_samples_for_Z=1000,
                 seed=None,
                 print_t=False):

        # Start timer
        start_time = time.time()

        # Record other inputs as class attributes
        self.alpha = alpha
        self.grid = grid
        self.grid_spacing = grid_spacing
        self.num_grid_points = num_grid_points
        self.bounding_box = bounding_box
        self.periodic = periodic
        self.Z_evaluation_method = evaluation_method_for_Z
        self.num_samples_for_Z = num_samples_for_Z
        self.t_start = t_start
        self.max_t_step = max_t_step
        self.print_t = print_t
        self.tolerance = tolerance
        self.seed = seed
        self.resolution = resolution
        self.num_posterior_samples = num_posterior_samples
        self.sample_only_at_l_star = sample_only_at_l_star
        self.max_log_evidence_ratio_drop = max_log_evidence_ratio_drop
        self.data = data
        self.compute_K_coefficient = compute_K_coeff
        self.results = None

        # Validate inputs
        self._inputs_check()

        # clean input data
        self._clean_data()

        # Choose grid
        self._set_grid()

        # Fit to data
        self._run()

        # Save some results
        self.histogram = self.results.R
        self.maxent = self.results.M
        self.phi_star_values = self.results.phi_star

        # Save K coefficient
        self.K_coeff = self.results.K_coeff

        # Compute evaluator for density
        self.density_func = Density(field_values=self.phi_star_values,
                                    grid=self.grid)

        # Compute optimal density at grid points
        self.values = self.evaluate(self.grid)

        # If any posterior samples were taken
        if num_posterior_samples > 0:

            # Save sampled phi values and weights
            self.sample_field_values = self.results.phi_samples
            self.sample_weights = self.results.phi_weights

            # Compute evaluator for all posterior samples
            self.sample_density_funcs = [
                Density(field_values=self.sample_field_values[:, k],
                        grid=self.grid)
                for k in range(self.num_posterior_samples)
            ]

            # Compute sampled values at grid points
            # These are NOT resampled
            self.sample_values = self.evaluate_samples(self.grid,
                                                       resample=False)

            # Compute effective sample size and efficiency
            self.effective_sample_size = np.sum(self.sample_weights)**2 \
                                        / np.sum(self.sample_weights**2)
            self.effective_sampling_efficiency = \
                self.effective_sample_size / self.num_posterior_samples

        # Store execution time in seconds
        self.runtime = time.time() - start_time

    @handle_errors
    def plot(self, ax=None,
             save_as=None,
             resample=True,
             figsize=(4, 4),
             fontsize=12,
             title='',
             xlabel='',
             tight_layout=False,
             show_now=True,
             show_map=True,
             map_color='blue',
             map_linewidth=2,
             map_alpha=1,
             num_posterior_samples=None,
             posterior_color='dodgerblue',
             posterior_linewidth=1,
             posterior_alpha=.2,
             show_histogram=True,
             histogram_color='orange',
             histogram_alpha=1,
             show_maxent=False,
             maxent_color='maroon',
             maxent_linewidth=1,
             maxent_alpha=1,
             backend='TkAgg'):
        """
        Plot the MAP density, the posterior sampled densities, and the
        data histogram.

        parameters
        ----------

        ax: (plt.Axes)
            A matplotlib axes object on which to draw. If None, one will be
            created

        save_as: (str)
            Name of file to save plot to. File type is determined by file
            extension.

        resample: (bool)
            If True, sampled densities will be ploted only after importance
            resampling.

        figsize: ([float, float])
            Figure size as (width, height) in inches.

        fontsize: (float)
            Size of font to use in plot annotation.

        title: (str)
            Plot title.

        xlabel: (str)
            Plot xlabel.

        tight_layout: (bool)
            Whether to call plt.tight_layout() after rendering graphics.

        show_now: (bool)
            Whether to show the plot immediately by calling plt.show().

        show_map: (bool)
            Whether to show the MAP density.

        map_color: (color spec)
            MAP density color.

        map_linewidth: (float)
            MAP density linewidth.

        map_alpha: (float)
            Map density opacity (between 0 and 1).

        num_posterior_samples: (int)
            Number of posterior samples to display. If this is greater than
            the number of posterior samples taken, all of the samples taken
            will be shown.

        posterior_color: (color spec)
            Sampled density color.

        posterior_linewidth: (float)
            Sampled density linewidth.

        posterior_alpha: (float)
            Sampled density opactity (between 0 and 1).

        show_histogram: (bool)
            Whether to show the (normalized) data histogram.

        histogram_color: (color spec)
            Face color of the data histogram.

        histogram_alpha: (float)
            Data histogram opacity (between 0 and 1).

        show_maxent: (bool)
            Whether to show the MaxEnt density estimate.

        maxent_color: (color spect)
            Line color of the MaxEnt density estimate.

        maxent_alpha: (float)
            MaxEnt opacity (between 0 and 1).

        backend: (str)
            Backend specification to send to sw.enable_graphics().

        returns
        -------

            None.

        """

        # check if matplotlib.pyplot is loaded. If not, load it carefully
        if 'matplotlib.pyplot' not in sys.modules:

            # First, enable graphics with the proper backend
            enable_graphics(backend=backend)

        # Make sure we have access to plt
        import matplotlib.pyplot as plt

        # If axes is not specified, create it and a corresponding figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            tight_layout = True

        # Plot histogram
        if show_histogram:
            ax.bar(self.grid,
                   self.histogram,
                   width=self.grid_spacing,
                   color=histogram_color,
                   alpha=histogram_alpha)

        # Plot maxent
        if show_maxent:
            ax.plot(self.grid,
                    self.maxent,
                    color=maxent_color,
                    linewidth=maxent_linewidth,
                    alpha=maxent_alpha)

        # Set number of posterior samples to plot
        if num_posterior_samples is None:
            num_posterior_samples = self.num_posterior_samples
        elif num_posterior_samples > self.num_posterior_samples:
            num_posterior_samples = self.num_posterior_samples

        # Plot posterior samples
        if num_posterior_samples > 0:
            sample_values = self.evaluate_samples(self.grid, resample=resample)
            ax.plot(self.grid,
                    sample_values[:, :num_posterior_samples],
                    color=posterior_color,
                    linewidth=posterior_linewidth,
                    alpha=posterior_alpha)

        # Plot best fit density
        if show_map:
            ax.plot(self.grid,
                    self.values,
                    color=map_color,
                    linewidth=map_linewidth,
                    alpha=map_alpha)

        # Style plot
        ax.set_xlim(self.bounding_box)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_yticks([])
        ax.tick_params('x', rotation=45, labelsize=fontsize)

        # Do not show interactive coordinates
        ax.format_coord = lambda x, y: ''

        # Do tight_layout if requested
        if tight_layout:
            plt.tight_layout()

        # Save figure if save_as is specified
        if save_as is not None:
            plt.draw()
            plt.savefig(save_as)

        # Show figure if show_now is True
        if show_now:
            plt.show()


    @handle_errors
    def evaluate(self, x):
        """
        Evaluate the optimal (i.e. MAP) density at the supplied value(s) of x.

        parameters
        ----------

        x: (number or list-like collection of numbers)
            The locations in the data domain at which to evaluate the MAP
            density.

        returns
        -------

        A float or 1D np.array representing the values of the MAP density at
        the specified locations.
        """

        # Clean input
        x_arr, is_number = clean_numerical_input(x)

        # Compute distribution values
        values = self.density_func.evaluate(x_arr)

        # If input is a single number, return a single number
        if is_number:
            values = values[0]

        # Return answer
        return values


    @handle_errors
    def evaluate_samples(self, x, resample=True):
        """
        Evaluate sampled densities at specified locations.

        parameters
        ----------

        x: (number or list-like collection of numbers)
            The locations in the data domain at which to evaluate sampled
            density.

        resample: (bool)
            Whether to use importance resampling, i.e., should the values
            returned be from the original samples (obtained using a Laplace
            approximated posterior) or should they be resampled to
            account for the deviation between the true Bayesian posterior
            and its Laplace approximation.

        returns
        -------

        A 1D np.array (if x is a number) or a 2D np.array (if x is list-like),
        representing the values of the posterior sampled densities at the
        specified locations. The first index corresponds to values in x, the
        second to sampled densities.
        """

        # Clean input
        x_arr, is_number = clean_numerical_input(x)

        # Check resample type
        check(isinstance(resample, bool),
              'type(resample) = %s. Must be bool.' % type(resample))

        # Make sure that posterior samples were taken
        check(self.num_posterior_samples > 0,
              'Cannot evaluate samples because no posterior samples'
              'have been computed.')

        assert(len(self.sample_density_funcs) == self.num_posterior_samples)

        # Evaluate all sampled densities at x
        values = np.array([d.evaluate(x_arr) for d
                           in self.sample_density_funcs]).T

        # If requested, resample columns of values array based on
        # sample weights
        if resample:
            probs = self.sample_weights / self.sample_weights.sum()
            old_cols = np.array(range(self.num_posterior_samples))
            new_cols = np.random.choice(old_cols,
                                        size=self.num_posterior_samples,
                                        replace=True,
                                        p=probs)
            values = values[:, new_cols]

        # If number was passed as input, return 1D np.array
        if is_number:
            values = values.ravel()

        return values

    @handle_errors
    def get_stats(self, use_weights=True,
                  show_samples=False):
        """
        Computes summary statistics for the estimated density

        parameters
        ----------

        show_samples: (bool)
            If True, summary stats are computed for each posterior sample.
            If False, summary stats are returned for the "star" estimate,
            the histogram, and the maxent estimate, along with the mean and
            RMSD values of these stats across posterior samples.

        use_weights: (bool)
            If True, mean and RMSD are computed using importance weights.

        returns
        -------

        df: (pd.DataFrame)
            A pandas data frame listing summary statistics for the estimated
            probability densities. These summary statistics include
            "entropy" (in bits), "mean", "variance", "skewness", and
            "kurtosis". If ``show_samples = False``, results will be shown for
            the best estimate, as well as mean and RMDS values across all
            samples. If ``show_samples = True``, results will be shown for
            each sample. A column showing column weights will also be included.
        """


        # Check inputs
        check(isinstance(use_weights, bool),
              'use_weights = %s; must be True or False.' % use_weights)
        check(isinstance(show_samples, bool),
              'show_samples = %s; must be True or False.' % show_samples)

        # Define a function for each summary statistic
        def entropy(Q):
            h = self.grid_spacing
            eps = 1E-10
            assert (all(Q >= 0))
            return -np.sum(h * Q * np.log2(Q + eps))

        def mean(Q):
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * x)

        def variance(Q):
            mu = mean(Q)
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * (x - mu) ** 2)

        def skewness(Q):
            mu = mean(Q)
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * (x - mu) ** 3) / np.sum(
                h * Q * (x - mu) ** 2) ** (3 / 2)

        def kurtosis(Q):
            mu = mean(Q)
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * (x - mu) ** 4) / np.sum(
                h * Q * (x - mu) ** 2) ** 2

        # Index functions by their names and set these as columns
        col2func_dict = {'entropy': entropy,
                         'mean': mean,
                         'variance': variance,
                         'skewness': skewness,
                         'kurtosis': kurtosis}
        cols = list(col2func_dict.keys())
        if show_samples:
            cols += ['weight']

        # Create list of row names
        if show_samples:
            rows = ['sample %d' % n
                     for n in range(self.num_posterior_samples)]
        else:
            rows = ['star', 'histogram', 'maxent',
                    'posterior mean', 'posterior RMSD']

        # Initialize data frame
        df = pd.DataFrame(columns=cols, index=rows)

        # Set sample weights
        if use_weights:
            ws = self.sample_weights
        else:
            ws = np.ones(self.num_posterior_samples)

        # Fill in data frame column by column
        for col_num, col in enumerate(cols):

            # If listing weights, do so
            if col == 'weight':
                df.loc[:, col] = ws

            # If computing a summary statistic
            else:

                # Get summary statistic function
                func = col2func_dict[col]

                # Compute func value for each sample
                ys = np.zeros(self.num_posterior_samples)
                for n in range(self.num_posterior_samples):
                    ys[n] = func(self.sample_values[:, n])

                # If recording individual results for all samples, do so
                if show_samples:
                    df.loc[:, col] = ys

                # Otherwise, record individual entries
                else:
                    # Fill in func value for start density
                    df.loc['star', col] = func(self.values)

                    # Fill in func value for histogram
                    df.loc['histogram', col] = func(self.histogram)

                    # Fill in func value for maxent point
                    df.loc['maxent', col] = func(self.maxent)

                    # Record mean and rmsd values across samples
                    mu = np.sum(ys * ws) / np.sum(ws)
                    df.loc['posterior mean', col] = mu
                    df.loc['posterior RMSD', col] = np.sqrt(
                        np.sum(ws * (ys - mu) ** 2) / np.sum(ws))

        # Return data frame to user
        return df

    def _run(self):
        """
        Estimates the probability density from data using the DEFT algorithm.
        Also samples posterior densities
        """

        # Extract information from Deft1D object
        data = self.data
        G = self.num_grid_points
        h = self.grid_spacing
        alpha = self.alpha
        periodic = self.periodic
        Z_eval = self.Z_evaluation_method
        num_Z_samples = self.num_samples_for_Z
        t_start = self.t_start
        DT_MAX = self.max_t_step
        print_t = self.print_t
        tollerance = self.tolerance
        resolution = self.resolution
        deft_seed = self.seed
        num_pt_samples = self.num_posterior_samples
        fix_t_at_t_star = self.sample_only_at_l_star
        max_log_evidence_ratio_drop = self.max_log_evidence_ratio_drop

        # Start clock
        start_time = time.time()

        # If deft_seed is specified, set it
        if not (deft_seed is None):
            np.random.seed(deft_seed)
        else:
            np.random.seed(None)

        # Create Laplacian
        laplacian_start_time = time.time()
        if periodic:
            op_type = '1d_periodic'
        else:
            op_type = '1d_bilateral'
        Delta = laplacian.Laplacian(op_type, alpha, G)
        laplacian_compute_time = time.time() - laplacian_start_time
        if print_t:
            print('Laplacian computed de novo in %f sec.'%laplacian_compute_time)

        # Get histogram counts and grid centers

        # Histogram based on bin centers
        counts, _ = np.histogram(data, self.bin_edges)
        N = sum(counts)

        # Make sure a sufficient number of bins are nonzero
        num_nonempty_bins = sum(counts > 0)
        check(num_nonempty_bins > self.alpha,
              'Histogram has %d nonempty bins; must be > %d.' %
              (num_nonempty_bins, self.alpha))

        # Compute initial t
        if t_start is None:
            t_start = min(0.0, sp.log(N)-2.0*alpha*sp.log(alpha/h))
            if t_start < -10.0:
                t_start /= 2
        if print_t:
            print('t_start = %0.2f' % t_start)

        # Do DEFT density estimation
        results = deft_core.run(counts, Delta, Z_eval, num_Z_samples,
                                t_start, DT_MAX, print_t,
                                tollerance, resolution, num_pt_samples,
                                fix_t_at_t_star,
                                max_log_evidence_ratio_drop,
                                self.compute_K_coefficient)

        # Normalize densities properly
        results.h = h
        results.L = G*h
        results.R /= h
        results.M /= h
        results.Q_star /= h
        results.l_star = h*(sp.exp(-results.t_star)*N)**(1/(2.*alpha))
        for p in results.map_curve.points:
            p.Q /= h
        if not (num_pt_samples == 0):
            results.Q_samples /= h
        results.Delta = Delta

        # Save evidence-related results
        points = results.map_curve.points
        self.ts = np.array([p.t for p in points])
        self.ells = h*(sp.exp(-self.ts)*N)**(1/(2.*alpha))
        self.log_Es = np.array([p.log_E for p in points])
        self.max_log_E = self.log_Es.max()

        # Store results
        self.results = results


    def _inputs_check(self):
        """
        Check all inputs NOT having to do with the choice of grid
        :param self:
        :return: None
        """

        if self.grid_spacing is not None:

            # max_t_step is a number
            check(isinstance(self.grid_spacing, numbers.Real),
                  'type(grid_spacing) = %s; must be a number' %
                  type(self.grid_spacing))

            # grid_spacing is positive
            check(self.grid_spacing > 0,
                  'grid_spacing = %f; must be > 0.' % self.grid_spacing)

        if self.grid is not None:

            # grid is a list or np.array
            types = (list, np.ndarray, np.matrix)
            check(isinstance(self.grid, types),
                  'type(grid) = %s; must be a list or np.ndarray' %
                  type(self.grid))

            # cast grid as np.array as ints
            try:
                self.grid = np.array(self.grid).ravel().astype(float)
            except: # SHOULD BE MORE SPECIFIC
                raise ControlledError('Cannot cast grid as 1D np.array of floats.')

            # grid has appropriate number of points
            check(2*self.alpha <= len(self.grid) <= MAX_NUM_GRID_POINTS,
                  'len(grid) = %d; must have %d <= len(grid) <= %d.' %
                  (len(self.grid), 2*self.alpha, MAX_NUM_GRID_POINTS))

            # grid is ordered
            diffs = np.diff(self.grid)
            check(all(diffs > 0),
                  'grid is not monotonically increasing.')

            # grid is evenly spaced
            check(all(np.isclose(diffs, diffs.mean())),
                  'grid is not evenly spaced; grid spacing = %f +- %f' %
                  (diffs.mean(), diffs.std()))

        # alpha is int
        check(isinstance(self.alpha, int),
              'type(alpha) = %s; must be int.' % type(self.alpha))

        # alpha in range
        check(1 <= self.alpha <= 4,
              'alpha = %d; must have 1 <= alpha <= 4' % self.alpha)

        if self.num_grid_points is not None:

            # num_grid_points is an integer
            check(isinstance(self.num_grid_points, int),
                  'type(num_grid_points) = %s; must be int.' %
                  type(self.num_grid_points))

            # num_grid_points is in the right range
            check(2*self.alpha <= self.num_grid_points <= MAX_NUM_GRID_POINTS,
              'num_grid_points = %d; must have %d <= num_grid_poitns <= %d.' %
              (self.num_grid_points, 2*self.alpha, MAX_NUM_GRID_POINTS))

        # bounding_box
        if self.bounding_box is not None:

            # bounding_box is right type
            box_types = (list, tuple, np.ndarray)
            check(isinstance(self.bounding_box, box_types),
                  'type(bounding_box) = %s; must be one of %s' %
                  (type(self.bounding_box), box_types))

            # bounding_box has right length
            check(len(self.bounding_box)==2,
                  'len(bounding_box) = %d; must be %d' %
                  (len(self.bounding_box), 2))

            # bounding_box entries must be numbers
            check(isinstance(self.bounding_box[0], numbers.Real) and
                  isinstance(self.bounding_box[1], numbers.Real),
                  'bounding_box = %s; entries must be numbers' %
                  repr(self.bounding_box))

            # bounding_box entries must be sorted
            check(self.bounding_box[0] < self.bounding_box[1],
                  'bounding_box = %s; entries must be sorted' %
                  repr(self.bounding_box))

            # reset bounding_box as tuple
            self.bounding_box = (float(self.bounding_box[0]),
                                 float(self.bounding_box[1]))

        # periodic is bool
        check(isinstance(self.periodic, bool),
              'type(periodic) = %s; must be bool' % type(self.periodic))

        # compute_K_coefficient is bool
        check(isinstance(self.compute_K_coefficient, bool),
              'type(compute_K_coefficient) = %s; must be bool' %
              type(self.periodic))

        # evaluation_method_for_Z is valid
        Z_evals = ['Lap', 'Lap+Imp', 'Lap+Fey']
        check(self.Z_evaluation_method in Z_evals,
              'Z_eval = %s; must be in %s' %
              (self.Z_evaluation_method, Z_evals))

        # num_samples_for_Z is an integer
        check(isinstance(self.num_samples_for_Z, numbers.Integral),
              'type(self.num_samples_for_Z) = %s; ' %
              type(self.num_samples_for_Z) +
              'must be integer.')
        self.num_samples_for_Z = int(self.num_samples_for_Z)

        # num_samples_for_Z is in range
        check(0 <= self.num_samples_for_Z <= MAX_NUM_SAMPLES_FOR_Z,
              'self.num_samples_for_Z = %d; ' % self.num_samples_for_Z +
              ' must satisfy 0 <= num_samples_for_Z <= %d.' %
               MAX_NUM_SAMPLES_FOR_Z)

        # max_t_step is a number
        check(isinstance(self.max_t_step, numbers.Real),
              'type(max_t_step) = %s; must be a number' %
              type(self.max_t_step))

        # max_t_step is positive
        check(self.max_t_step > 0,
              'maxt_t_step = %f; must be > 0.' % self.max_t_step)

        # print_t is bool
        check(isinstance(self.print_t,bool),
              'type(print_t) = %s; must be bool.' % type(self.print_t))

        # tolerance is float
        check(isinstance(self.tolerance, numbers.Real),
              'type(tolerance) = %s; must be number' % type(self.tolerance))

        # tolerance is positive
        check(self.tolerance > 0,
              'tolerance = %f; must be > 0' % self.tolerance)

        # resolution is number
        check(isinstance(self.resolution, numbers.Real),
              'type(resolution) = %s; must be number' % type(self.resolution))

        # resolution is positive
        check(self.resolution > 0,
              'resolution = %f; must be > 0' % self.resolution)

        if self.seed is not None:

            # seed is int
            check(isinstance(self.seed, int),
                  'type(seed) = %s; must be int' % type(self.seed))

            # seed is in range
            check(0 <= self.seed <= 2**32 - 1,
                  'seed = %d; must have 0 <= seed <= 2**32 - 1' % self.seed)

        # sample_only_at_l_star is bool
        check(isinstance(self.sample_only_at_l_star, bool),
              'type(sample_only_at_l_star) = %s; must be bool.' %
              type(self.sample_only_at_l_star))

        # num_posterior_samples is int
        check(isinstance(self.num_posterior_samples, numbers.Integral),
              'type(num_posterior_samples) = %s; must be integer' %
              type(self.num_posterior_samples))
        self.num_posterior_samples = int(self.num_posterior_samples)


        # num_posterior_samples is nonnegative
        check(0 <= self.num_posterior_samples <= MAX_NUM_POSTERIOR_SAMPLES,
              'num_posterior_samples = %f; need '%self.num_posterior_samples +
              '0 <= num_posterior_samples <= %d.' %MAX_NUM_POSTERIOR_SAMPLES)

        # max_log_evidence_ratio_drop is number
        check(isinstance(self.max_log_evidence_ratio_drop, numbers.Real),
              'type(max_log_evidence_ratio_drop) = %s; must be number' %
              type(self.max_log_evidence_ratio_drop))

        # max_log_evidence_ratio_drop is positive
        check(self.max_log_evidence_ratio_drop > 0,
              'max_log_evidence_ratio_drop = %f; must be > 0' %
              self.max_log_evidence_ratio_drop)


    def _clean_data(self):
        """
        Sanitize the assigned data
        :param: self
        :return: None
        """
        data = self.data

        # if data is a list-like, convert to 1D np.array
        if isinstance(data, LISTLIKE):
            data = np.array(data).ravel()
        elif isinstance(data, set):
            data = np.array(list(data)).ravel()
        else:
            raise ControlledError(
                "Error: could not cast data into an np.array")

        # Check that entries are numbers
        check(all([isinstance(n, numbers.Real) for n in data]),
              'not all entries in data are real numbers')

        # Cast as 1D np.array of floats
        data = data.astype(float)

        # Keep only finite numbers
        data = data[np.isfinite(data)]


        try:
            if not (len(data) > 0):
                raise ControlledError(
                    'Input check failed, data must have length > 0: data = %s' % data)
        except ControlledError as e:
            print(e)
            sys.exit(1)

        try:
            data_spread = max(data) - min(data)
            if not np.isfinite(data_spread):
                raise ControlledError(
                    'Input check failed. Data[max]-Data[min] is not finite: Data spread = %s' % data_spread)
        except ControlledError as e:
            print(e)
            sys.exit(1)

        try:
            if not (data_spread > 0):
                raise ControlledError(
                    'Input check failed. Data[max]-Data[min] must be > 0: data_spread = %s' % data_spread)
        except ControlledError as e:
            print(e)
            sys.exit(1)

        # Set cleaned data
        self.data = data


    def _set_grid(self):
        """
        Sets the grid based on user input
        """

        data = self.data
        grid = self.grid
        grid_spacing = self.grid_spacing
        num_grid_points = self.num_grid_points
        bounding_box = self.bounding_box
        alpha = self.alpha

        # If grid is specified
        if grid is not None:

            # Check and set number of grid points
            num_grid_points = len(grid)
            assert(num_grid_points >= 2*alpha)

            # Check and set grid spacing
            diffs = np.diff(grid)
            grid_spacing = diffs.mean()
            assert (grid_spacing > 0)
            assert (all(np.isclose(diffs, grid_spacing)))

            # Check and set grid bounds
            grid_padding = grid_spacing / 2
            lower_bound = grid[0] - grid_padding
            upper_bound = grid[-1] + grid_padding
            bounding_box = np.array([lower_bound, upper_bound])
            box_size = upper_bound - lower_bound

        # If grid is not specified
        if grid is None:

            ### First, set bounding box ###

            # If bounding box is specified, use that.
            if bounding_box is not None:
                assert bounding_box[0] < bounding_box[1]
                lower_bound = bounding_box[0]
                upper_bound = bounding_box[1]
                box_size = upper_bound - lower_bound


            # Otherwise set bounding box based on data
            else:
                assert isinstance(data, np.ndarray)
                assert all(np.isfinite(data))
                assert min(data) < max(data)

                # Choose bounding box to encapsulate all data, with extra room
                data_max = max(data)
                data_min = min(data)
                data_span = data_max - data_min
                lower_bound = data_min - .2 * data_span
                upper_bound = data_max + .2 * data_span

                # Autoadjust lower bound
                if data_min >= 0 and lower_bound < 0:
                    lower_bound = 0

                # Autoadjust upper bound
                if data_max <= 0 and upper_bound > 0:
                    upper_bound = 0
                if data_max <= 1 and upper_bound > 1:
                    upper_bound = 1
                if data_max <= 100 and upper_bound > 100:
                    upper_bound = 100

                # Extend bounding box outward a little for numerical safety
                lower_bound -= SMALL_NUM*data_span
                upper_bound += SMALL_NUM*data_span
                box_size = upper_bound - lower_bound

                # Set bounding box
                bounding_box = np.array([lower_bound, upper_bound])

            ### Next, define grid based on bounding box ###

            # If grid_spacing is specified
            if (grid_spacing is not None):
                assert isinstance(grid_spacing, float)
                assert np.isfinite(grid_spacing)
                assert grid_spacing > 0

                # Set number of grid points
                num_grid_points = np.floor(box_size/grid_spacing).astype(int)

                # Check num_grid_points isn't too small
                check(2*self.alpha <= num_grid_points,
                      'Using grid_spacing = %f ' % grid_spacing +
                      'produces num_grid_points = %d, ' % num_grid_points +
                      'which is too small. Reduce grid_spacing or do not set.')

                # Check num_grid_points isn't too large
                check(num_grid_points <= MAX_NUM_GRID_POINTS,
                      'Using grid_spacing = %f ' % grid_spacing +
                      'produces num_grid_points = %d, ' % num_grid_points +
                      'which is too big. Increase grid_spacing or do not set.')

                # Define grid padding
                # Note: grid_spacing/2 <= grid_padding < grid_spacing
                grid_padding = (box_size - (num_grid_points-1)*grid_spacing)/2
                assert (grid_spacing/2 <= grid_padding < grid_spacing)

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM), # For safety
                                   num_grid_points)

            # Otherwise, if num_grid_points is specified
            elif (num_grid_points is not None):
                assert isinstance(num_grid_points, int)
                assert 2*alpha <= num_grid_points <= MAX_NUM_GRID_POINTS

                # Set grid spacing
                grid_spacing = box_size / num_grid_points

                # Define grid padding
                grid_padding = grid_spacing/2

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM), # For safety
                                   num_grid_points)

            # Otherwise, set grid_spacing and num_grid_points based on data
            else:
                assert isinstance(data, np.ndarray)
                assert all(np.isfinite(data))
                assert min(data) < max(data)

                # Compute default grid spacing
                default_grid_spacing = box_size/DEFAULT_NUM_GRID_POINTS

                # Set minimum number of grid points
                min_num_grid_points = 2 * alpha

                # Set minimum grid spacing
                data.sort()
                diffs = np.diff(data)
                min_grid_spacing = min(diffs[diffs > 0])
                min_grid_spacing = min(min_grid_spacing,
                                       box_size/min_num_grid_points)

                # Set grid_spacing
                grid_spacing = max(min_grid_spacing, default_grid_spacing)

                # Set number of grid points
                num_grid_points = np.floor(box_size/grid_spacing).astype(int)

                # Set grid padding
                grid_padding = grid_spacing/2

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM),  # For safety
                                   num_grid_points)

        # Set final grid
        self.grid = grid
        self.grid_spacing = grid_spacing
        self.grid_padding = grid_padding
        self.num_grid_points = num_grid_points
        self.bounding_box = bounding_box
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.box_size = box_size

        # Make sure that the final number of gridpoints is ok.
        check(2 * self.alpha <= self.num_grid_points <= MAX_NUM_GRID_POINTS,
              'After setting grid, we find that num_grid_points = %d; must have %d <= len(grid) <= %d. ' %
              (self.num_grid_points, 2*self.alpha, MAX_NUM_GRID_POINTS) +
              'Something is wrong with input values of grid, grid_spacing, num_grid_points, or bounding_box.')

        # Set bin edges
        self.bin_edges = np.concatenate(([lower_bound],
                                         grid[:-1]+grid_spacing/2,
                                         [upper_bound]))

