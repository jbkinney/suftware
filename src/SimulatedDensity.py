import numpy as np
from suftware.src.GaussianMixtureDensity import GaussianMixtureDensity
from suftware.src.ScipyStatsDensity import ScipyStatsDensity
from suftware.src.utils import check, handle_errors
import scipy.stats as ss

@handle_errors
def SimulatedDensity(name):
    """
    This is actually a function, not a class.
    Returns an pre-defined Density object with the specified name.
    """
    check(name in density_dict.keys(),
          "Example density name '%s' is invalid. "
          "Please choose a valid name from: "
          "\n\t%s" % (name, '\n\t'.join(density_names)))

    return density_dict[name]

@handle_errors
def SimulatedDataset(name='DoubleGaussian', num_data_points=100, seed=None):
    """
    Simulates data from a variety of distributions.

    parameters
    ----------

    distribution: (str)
        The distribution from which to draw data. Run sw.SimulatedDataset.list()
        to which distributions are available.

    num_data_points: (int > 0)
        The number of data points to simulate. Must satisfy
        0 <= N <= MAX_DATASET_SIZE.

    seed: (int)
        Seed passed to random number generator.

    """

    density = SimulatedDensity(name)
    dataset = density.sample(num_samples=num_data_points, seed=seed)
    return dataset

@handle_errors
def list_simulated_densities():
    """ Returns a list of valid example density names"""
    return density_names.copy()


#
# Create list of simulated densities below
#

density_list = []

# Add Gaussian mixture densities
density_list.append(
    GaussianMixtureDensity(name='DoubleGaussian',
                           bounding_box=[-15, 15],
                           weights=[2, 1],
                           mus=[-2, 2],
                           sigmas=[1, 1]))
density_list.append(
    GaussianMixtureDensity(name='DoubleGaussianZoom',
                           bounding_box=[-3, 3],
                           weights=[2, 1],
                           mus=[-2, 2],
                           sigmas=[1, 1]))
density_list.append(
    GaussianMixtureDensity(name='Normal',
                           bounding_box=[-5,5],
                           mus=[0],
                           sigmas=[1],
                           weights=[1]))
density_list.append(
    GaussianMixtureDensity(name='GM_Narrow',
                           bounding_box=[-6, 6],
                           weights=[1, 1],
                           mus=[-1.25, 1.25],
                           sigmas=[1, 1]))
density_list.append(
    GaussianMixtureDensity(name='GM_Wide',
                           bounding_box=[-6, 6],
                           weights=[1, 1],
                           mus=[-2, 2],
                           sigmas=[1, 1]))
density_list.append(
    GaussianMixtureDensity(name='GM_Foothills',
                           bounding_box=[-5,12],
                           weights=[1., 1., 1., 1., 1.],
                           mus=[0., 5., 8., 10, 11],
                           sigmas=[2., 1., 0.5, 0.25, 0.125]))
density_list.append(
    GaussianMixtureDensity(name='GM_Accordian',
                           bounding_box=[-5, 13],
                           weights=[16., 8., 4., 2., 1., 0.5],
                           mus=[0., 5., 8., 10, 11, 11.5],
                           sigmas=[2., 1., 0.5, 0.25, 0.125, 0.0625]))
density_list.append(
    GaussianMixtureDensity(name='GM_Goalposts',
                           bounding_box=[-25, 25],
                           weights=[1, 1],
                           mus=[-20, 20],
                           sigmas=[1, 1]))
density_list.append(
    GaussianMixtureDensity(name='GM_Goalposts',
                         bounding_box=[-25, 25],
                         weights=[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                         mus=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
                         sigmas=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

# Add scipy.stats densities
density_list.append(
    ScipyStatsDensity(name='Argus',
                      ss_obj=ss.argus(chi=.1),
                      bounding_box=[0, 1]))
density_list.append(
    ScipyStatsDensity(name='Beta',
                      ss_obj=ss.beta(a=1.5, b=5),
                      bounding_box=[0, 1]))
density_list.append(
    ScipyStatsDensity(name='Betaprime',
                      ss_obj=ss.betaprime(a=5, b=6),
                      bounding_box=[0, 4]))
density_list.append(
    ScipyStatsDensity(name='Bradford',
                      ss_obj=ss.bradford(c=3),
                      bounding_box=[0, 1]))
density_list.append(
    ScipyStatsDensity(name='Cauchy',
                      ss_obj=ss.cauchy(),
                      bounding_box=[-2, 2]))
density_list.append(
    ScipyStatsDensity(name='ChiSquare',
                      ss_obj=ss.chi2(df=10),
                      bounding_box=[0, 20]))
density_list.append(
    ScipyStatsDensity(name='ExpNormal',
                      ss_obj=ss.exponnorm(K=3),
                      bounding_box=[-3, 5]))
density_list.append(
    ScipyStatsDensity(name='ExpPower',
                      ss_obj=ss.exponpow(b=1.5),
                      bounding_box=[0, 2]))
density_list.append(
    ScipyStatsDensity(name='ExpWeibull',
                      ss_obj=ss.exponweib(a=1.5, c=1.5),
                      bounding_box=[0, 3]))
density_list.append(
    ScipyStatsDensity(name='FoldedNormal',
                      ss_obj=ss.foldnorm(c=2),
                      bounding_box=[0, 5]))
density_list.append(
    ScipyStatsDensity(name='Gamma',
                      ss_obj=ss.gamma(a=3),
                      bounding_box=[0, 10]))
density_list.append(
    ScipyStatsDensity(name='Pareto',
                      ss_obj=ss.pareto(b=4),
                      bounding_box=[1, 4]))
density_list.append(
    ScipyStatsDensity(name='Semicircular',
                      ss_obj=ss.semicircular(),
                      bounding_box=[-2, 2]))
density_list.append(
    ScipyStatsDensity(name='SkewNorm',
                      ss_obj=ss.skewnorm(a=5),
                      bounding_box=[-1, 4]))
density_list.append(
    ScipyStatsDensity(name='vonMises',
                      ss_obj=ss.vonmises(kappa=.5),
                      bounding_box=[0, 4]))
density_list.append(
    ScipyStatsDensity(name='Wald',
                      ss_obj=ss.wald(),
                      bounding_box=[0, 1.5]))
density_list.append(
    ScipyStatsDensity(name='WeibullMin',
                      ss_obj=ss.weibull_min(c=2),
                      bounding_box=[0, 5]))
density_list.append(
    ScipyStatsDensity(name='WrapCauchy',
                      ss_obj=ss.wrapcauchy(c=.4),
                      bounding_box=[0, 2*np.pi]))

# Convert list to dictionary
density_names = [d.name for d in density_list]
density_dict = dict(zip(density_names, density_list))
density_names.sort()
