# DON'T RUN - work in progress - unchecked consequences!
#
# DATA BELOW IS NOT VALID

from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='suftware-refactored',
      version='1.1.1',
      description='Refactored Statistics Using Field Theory',
      long_description=readme(),
      classifiers=[
        'Development Status :: ',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      keywords='density estimation',
      url='http://github.com/jbkinney/suftware???',
      author='Wei-Chia Chen, Ammar Tareen, Justin B. Kinney et al.',
      author_email='',
      license='MIT',
      packages=['suftware'],
      package_data={'suftware': ['suftware_data/*']},
      include_package_data=True,
      install_requires=[
          'scipy',
          'numpy',
          'matplotlib',
      ],
      test_suite='nose.collector',
	  tests_require=['nose'],
      zip_safe=False)