"""
Head module. Contains classes for user interfacing.
"""

# Make classes local
from .density_estimator import DensityEstimator
from .example_dataset import ExampleDataset
from .density import Density
from .gaussian_mixture_density import GaussianMixtureDensity
from .utils import check, ControlledError
from .simulations import simulate_density, simulate_dataset
from .simulations import list_simulated_densities
from .example_dataset import list_example_datasets
from .scipy_stats_density import ScipyStatsDensity


def demo(example='real_data'):
    """
    Performs a demonstration of suftware.

    Parameters
    ----------

    example: (str)
        A string specifying which demo to run. Must be 'real_data' or
        'simulated_data'.

    Return
    ------

    None.
    """

    import os
    example_dir = os.path.dirname(__file__)

    example_dict = {
        'simulated_data': 'examples/example_wide.py',
        'real_data': 'examples/example_alcohol.py'
    }

    check(example in example_dict,
          'example = %s is not valid. Must be one of %s'%\
          (example, example_dict.keys()))

    file_name = '%s/%s'%(example_dir, example_dict[example])
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s'%\
              (file_name, line, content, line))
    exec(open(file_name).read())
