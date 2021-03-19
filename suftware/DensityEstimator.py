import sys
import numbers

import numpy as np
import scipy as sp

# Import deft-related code

import deft_core
import laplacian
from densityevaluator import InterpolatedPDF


SMALL_NUM = 1E-6  #? arbitrary, non-coherent with the `utils` module.
MAX_NUM_GRID_POINTS = 1000 #? what for? I have BIG COMPUTER - and...
DEFAULT_NUM_GRID_POINTS = 100
MAX_NUM_POSTERIOR_SAMPLES = 1000
MAX_NUM_SAMPLES_FOR_Z = 1000



class SimpleDensityEstimator:
    """
    """

    def __init__(self, data,
                 npts=DEFAULT_NUM_GRID_POINTS,
                 alpha=3,
                 periodic=False,
                 max_t_step=1.0,
                 tolerance=1E-6,
                 resolution=0.1,
                 sample_only_at_l_star=False,
                 max_log_evidence_ratio_drop=20,
                 evaluation_method_for_Z='Lap',
                 num_samples_for_Z=1000):

        # Record other inputs as class attributes
        #
        self.alpha = alpha
        self.grid = None
        self.step = None
        self.npts = npts
        self.low = None
        self.high = None
        self.bounding_box = bounding_box
        self.periodic = periodic
        self.Z_evaluation_method = evaluation_method_for_Z
        self.num_samples_for_Z = num_samples_for_Z
        self.max_t_step = max_t_step
        self.print_t = print_t
        self.tolerance = tolerance
        self.resolution = resolution
        self.sample_only_at_l_star = sample_only_at_l_star
        self.max_log_evidence_ratio_drop = max_log_evidence_ratio_drop
        self.data = data
        self.results = None

        # init_bbox
        #
        PADDING_FACTOR = 0.2
        x1 = np.min(data)
        x2 = np.max(data)
        self.low = x1 - (x2 - x1) * PADDING_FACTOR
        self.high = x2 + (x2 - x1) * PADDING_FACTOR
        if self.low < 0 and x1 >= 0:
            self.low = 0
        if self.high > 0 and x2 > 0:
            self.high = 0

        # Define a grid based on self.low and self.high (i.e. based on the
        # bounding box): set grid spacing, define grid padding, define grid to
        # be centered in bounding box, set bin edges.
        #
        self.step = (self.high - self.low) / self.npts
        self.grid_padding = self.step / 2
        grid_start = self.low + self.grid_padding
        grid_stop = self.high - self.grid_padding
        self.grid = np.linspace(
            grid_start, grid_stop * (1 + SMALL_NUM), # For safety?!?!? WTF?!
            self.npts)
        self.bin_edges = np.concatenate(
            ([self.low], self.grid[:-1] + self.step / 2, [self.high]))


        # Fit to data
        self._run()

        #? WTF, the results are already avaliable as self.results.*
        #?
        # Save some results
        #
        self.histogram = self.results.R
        self.maxent = self.results.M
        self.phi_star_values = self.results.phi_star

        # Compute evaluator for density
        #
        self.pdf = InterpolatedPDF(
            self.results.phi_star, self.grid, self.low, self.high)

    @handle_errors
    def get_stats(self, use_weights=True, show_samples=False):
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


        # Define a function for each summary statistic
        def entropy(Q):
            h = self.step
            eps = 1E-10
            assert (all(Q >= 0))
            return np.sum(h * Q * np.log2(Q + eps))

        def mean(Q):
            x = self.grid
            h = self.step
            return np.sum(h * Q * x)

        def variance(Q):
            mu = mean(Q)
            x = self.grid
            h = self.step
            return np.sum(h * Q * (x - mu) ** 2)

        def skewness(Q):
            mu = mean(Q)
            x = self.grid
            h = self.step
            return np.sum(h * Q * (x - mu) ** 3) / np.sum(
                h * Q * (x - mu) ** 2) ** (3 / 2)

        def kurtosis(Q):
            mu = mean(Q)
            x = self.grid
            h = self.step
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
        """

        #? G == self.npts # just remove G-something

        laplace_operator = laplacian.Laplacian(self.periodic, self.alpha, 
                self.npts)

        # Get histogram counts and grid centers

        # Histogram based on bin centers
        counts, _ = np.histogram(data, self.bin_edges)
        counts_total = sum(counts)

        # Compute initial t
        #
        t_start = min(0.0,
                      np.log(counts_total)
                      - 2 * self.alpha * np.log(self.alpha / self.step))

        # Do DEFT density estimation
        #
        self.results = deft_core.run(
            counts,
            laplace_operator,
            Z_eval,
         num_Z_samples,
    self.npts       t_start,
            print_t,
            tollerance,
            resolution,
            num_pt_samples,
            fix_t_at_t_star,
            max_log_evidence_ratio_drop)

        # Normalize densities properly
        self.results.step = self.step
        self.results.L = self.npts * self.step
        self.results.R /= self.step
        self.results.M /= self.step
        self.results.Q_star /= self.step
        self.results.l_star = self.step * (
            sp.exp(-self.results.t_star) * counts_total)**(1/(2.*self.alpha))
        for p in self.results.map_curve.points:
            p.Q /= self.step
        if not (num_pt_samples == 0):
            results.Q_samples /= self.step
