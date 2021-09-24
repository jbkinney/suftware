#!/usr/bin/env python
"""
Functional Tests for SUFTware
"""

# Standard modules
import numpy as np
import sys
import time

# Import suftware 
sys.path.append('../')
import suftware as sw

# Generate data
np.random.seed(0)
data = np.random.randn(100)

# Generate density so methods can be tested
density = sw.DensityEstimator(data)

global_success_counter = 0
global_fail_counter = 0

# Common success and fail lists
bool_fail_list = [0, -1, 'True', 'x', 1]
bool_success_list = [False, True]

# helper method for functional test_for_mistake
def test_for_mistake(func, *args, **kw):
    """
    Run a function with the specified parameters and register whether
    success or failure was a mistake

    parameters
    ----------

    func: (function or class constructor)
        An executable function to which *args and **kwargs are passed.

    return
    ------

    None.
    """
    global global_fail_counter
    global global_success_counter

    # print test number
    test_num = global_fail_counter + global_success_counter
    print('Test # %d: ' % test_num, end='')

    # Run function
    obj = func(*args, **kw)

    # Increment appropriate counter
    if obj.mistake:
        global_fail_counter += 1
    else:
        global_success_counter += 1


def test_parameter_values(func,
                          var_name=None,
                          fail_list=[],
                          success_list=[],
                          **kwargs):
    """
    Tests predictable success & failure of different values for a
    specified parameter when passed to a specified function

    parameters
    ----------

    func: (function)
        Executable to test. Can be function or class constructor.

    var_name: (str)
        Name of variable to test. If not specified, function is
        tested for success in the absence of any passed parameters.

    fail_list: (list)
        List of values for specified variable that should fail

    success_list: (list)
        List of values for specified variable that should succeed

    **kwargs:
        Other keyword variables to pass onto func.

    return
    ------

    None.

    """

    # If variable name is specified, test each value in fail_list
    # and success_list
    if var_name is not None:

        # User feedback
        print("Testing %s() parameter %s ..." % (func.__name__, var_name))

        # Test parameter values that should fail
        for x in fail_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=True, **kwargs)

        # Test parameter values that should succeed
        for x in success_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=False, **kwargs)

        print("Tests passed: %d. Tests failed: %d.\n" %
              (global_success_counter, global_fail_counter))

    # Otherwise, make sure function without parameters succeeds
    else:

        # User feedback
        print("Testing %s() without parameters." % func.__name__)

        # Test function
        test_for_mistake(func=func, should_fail=False, **kwargs)


def test_SimulatedDataset___init__():
    """
    Test SimulatedDataset()
    """

    # No parameters
    test_parameter_values(func=sw.SimulatedDataset)

    # dataset
    test_parameter_values(
        func=sw.SimulatedDataset,
        var_name='distribution',
        fail_list=[
            None,
            1,
            [],
            'bogus_distribution'
        ],
        success_list=sw.SimulatedDataset.list()
    )

    # num_data_points
    test_parameter_values(
        func=sw.SimulatedDataset,
        var_name='num_data_points',
        fail_list=[
            None,
            -1,
            'nonsense',
            0,
            100.0,
            1E6,
            int(1E6)+1,
        ],
        success_list=[
            1,
            100,
            int(1E6),
        ]
    )

    # seed
    test_parameter_values(
        func=sw.SimulatedDataset,
        var_name='seed',
        fail_list=[
            -1,
            'nonsense',
            100.0,
            2**32,
        ],
        success_list=[
            None,
            0,
            1,
            1000,
            2**32-1
        ]
    )

def test_SimulatedDataset_list():
    """
    Test SimulatedDataset.list()
    """

    # No parameters
    test_parameter_values(func=sw.SimulatedDataset.list)




def test_ExampleDataset___init__():
    """
    Test ExampleDataset()
    """

    # No parameters
    test_parameter_values(func=sw.ExampleDataset)

    # dataset
    test_parameter_values(
        func=sw.ExampleDataset,
        var_name='dataset',
        fail_list=[
            None,
            1,
            [],
            'nonexistent_dataset',
            'who.',
        ],
        success_list=sw.ExampleDataset.list()
    )


def test_ExampleDensity_list():
    """
    Test SimulatedDensity.list()
    """
    # No parameters
    test_parameter_values(func=sw.ExampleDataset.list)


def test_DensityEstimator_evaluate_samples():
    """
    Test DensityEstimator.evaluate_sample()
    """

    # x
    test_parameter_values(
        func=density.evaluate_samples,
        var_name='x',
        fail_list=[
            None,
            '1.0',
            1+2j,
            np.nan,
            np.Inf,
            {1:1, 2:2}.keys(),
            {1:1, 2:2}.values()
        ],
        success_list=[
            0,
            -1,
            1,
            1E6,
            range(10),
            np.random.randn(10),
            np.random.randn(3, 3),
            np.matrix(range(10)),
            np.matrix(range(10)).T,
            np.random.randn(2, 2, 2, 2)
        ]
    )

    # resample
    test_parameter_values(
        func=density.evaluate_samples,
        var_name='resample',
        fail_list=bool_fail_list,
        success_list=bool_success_list,
        x=density.grid
    )


def test_DensityEstimator_evaluate():
    """
    Test DensityEstimator.evaluate()
    """

    # x
    test_parameter_values(
        func=density.evaluate,
        var_name='x',
        fail_list=[
            None,
            '1.0',
            1+2j,
            np.nan,
            np.Inf,
            {1:1, 2:2}.keys(),
            {1:1, 2:2}.values()
        ],
        success_list=[
            0,
            -1,
            1,
            1E6,
            range(10),
            np.random.randn(10),
            np.random.randn(3, 3),
            np.matrix(range(10)),
            np.matrix(range(10)).T,
            np.random.randn(2, 2, 2, 2)
        ]
    )


def test_DensityEstimator_get_stats():
    """
    Test DensityEstimator.get_stats()
    """

    # use_weights
    test_parameter_values(
        func=density.get_stats,
        var_name='use_weights',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )

    # show_samples
    test_parameter_values(
        func=density.get_stats,
        var_name='show_samples',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )



def test_DensityEstimator___init__():
    """
    Test DensityEstimator()
    """

    # data
    test_parameter_values(
        func=sw.DensityEstimator,
        var_name='data',
        fail_list=[
            None,
            5,
            [str(x) for x in data],
            [1]*5 + [2]*10,
            data.astype(complex)
        ],
        success_list=[
            data,
            np.random.randn(10),
            np.random.randn(int(1E6)),
            range(100),
            list(data),
            list(data) + [np.nan, np.Inf, -np.Inf],
            set(data)
        ]
    )

    # grid
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='grid',
        fail_list=[
            5,
            'x',
            set(np.linspace(-3, 3, 100)),
            np.linspace(-3, 3, 5),
            np.linspace(-3, 3, 1001),
            np.linspace(-1E-6, 1E-6, 100),
            np.linspace(-1E6, 1E6, 100)
        ],
        success_list=[
            None,
            np.linspace(-3, 3, 100),
            np.linspace(-3, 3, 100).T,
            np.matrix(np.linspace(-3, 3, 100)),
            np.matrix(np.linspace(-3, 3, 100).T),
            list(np.linspace(-3, 3, 100)),
            np.linspace(-3, 3, 6),
            np.linspace(-3, 3, 100),
            np.linspace(-3, 3, 100),
            np.linspace(-3, 3, 1000)
        ]
    )

    # grid_spacing
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='grid_spacing',
        fail_list=[
            0,
            0.0,
            -0.1,
            '0.1',
            [0.1],
            0.0001,
            1000.0
        ],
        success_list=[
            None,
            0.05,
            0.1,
            0.5
        ]
    )

    # bounding_box
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='bounding_box',
        fail_list=[
            {-6, 6},
            6,
            [6],
            [-6, 0, 6],
            ['-6', '6'],
            [6, 6],
            [-1E-6, 1E-6],
            [-1E6, 1E6],
            [10, 20]
        ],
        success_list=[
            [-6, 6],
            (-6, 6),
            np.array([-6, 6]),
            [-.1, .1],
            [-10, 10]
        ]
    )

    # num_grid_points
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='num_grid_points',
        fail_list=[-10, -1, 0, 1, 2, 3, 4, 5, 1001],
        success_list=[6, 100, 1000]
    )

    # alpha
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='alpha',
        fail_list=[None, 'x', -1, 0.0, 0, 0.1, 10],
        success_list=[1, 2, 3, 4]
    )

    # periodic
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='periodic',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )

    # evaluation_method_for_Z
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='evaluation_method_for_Z',
        fail_list=[0, 'x', 'Einstein', False],
        success_list=['Lap', 'Lap+Fey', 'Lap+Imp']
    )

    # num_samples_for_Z
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='num_samples_for_Z',
        fail_list=[None, -1, 'x', 0.1, 1001],
        success_list=[0, 1, 10, 1000]
    )

    # tolerance
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='tolerance',
        fail_list=['x', -1, 0, 0.0],
        success_list=[1e-6, 1e-4, 1e-2, 1e-1, 1]
    )

    # resolution
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='resolution',
        fail_list=['x', -1, 0, 0.0, None],
        success_list = [1e-4, 1e-2, 1e-1, 1]
    )

    # seed
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='seed',
        fail_list=['x', 1e-5, 1.0, -1],
        success_list=[None, 1, 10, 100, 1000]
    )

    # print_t
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='print_t',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )

    # num_posterior_samples
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='num_posterior_samples',
        fail_list=['x', -1, 0.0, 1001],
        success_list=[0, 1, 2, 3, 10, 100, 1000]
    )

    # sample_only_at_l_star
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='sample_only_at_l_star',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )

    # max_log_evidence_ratio_drop
    test_parameter_values(
        func=sw.DensityEstimator,
        data=data,
        var_name='max_log_evidence_ratio_drop',
        fail_list=['x', -1, 0, None, 0],
        success_list=[0.1, 1, 2, 3, 10, 100, 100.0, 1000]
    )


# Run functional tests
if __name__ == '__main__':

    # Start timer
    start_time = time.time()

    # DensityEstimator methods
    test_DensityEstimator___init__()
    test_DensityEstimator_get_stats()
    test_DensityEstimator_evaluate()
    test_DensityEstimator_evaluate_samples()

    # SimulatedDensity methods
    test_ExampleDataset___init__()
    test_ExampleDensity_list()

    # SimulateDensity methods
    test_SimulatedDataset___init__()
    test_SimulatedDataset_list()

    # Compute time
    t = time.time() - start_time
    t_min = int(t/60)
    t_sec = int(t)%60

    # Print results
    print("\nFunctional tests took %d min %d sec." % (t_min, t_sec))
    print("Total tests passed: %d. Total tests failed: %d.\n" %
          (global_success_counter, global_fail_counter))
