"""
Head module. Contains classes for user interfacing.
"""

# Make classes local
from suftware.src.DensityEstimator import DensityEstimator
from suftware.src.ExampleDataset import ExampleDataset
from suftware.src.Density import Density
from suftware.src.GaussianMixtureDensity import GaussianMixtureDensity
from suftware.src.utils import ControlledError as ControlledError
from suftware.src.SimulatedDensity import SimulatedDensity, SimulatedDataset
from suftware.src.SimulatedDensity import list_simulated_densities
from suftware.src.ExampleDataset import list_example_datasets
from suftware.src.ScipyStatsDensity import ScipyStatsDensity


# Enable plotting
from suftware.src.utils import enable_graphics, check, ControlledError


def demo(example='real_data'):
    """
    Performs a demonstration of suftware.

    Parameters
    ----------

    example: (str)
        A string specifying which demo to run. Must be 'real_data',
        'simulated_data', or 'custom_data'.

    Return
    ------

    None.
    """

    import os
    example_dir = os.path.dirname(__file__)

    example_dict = {
        'custom_data': 'docs/example_custom.py',
        'simulated_data': 'docs/example_wide.py',
        'real_data': 'docs/example_alcohol.py'
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
