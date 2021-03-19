========
SUFTware
========

*Written by Wei-Chia Chen, Ammar Tareen, and Justin B. Kinney.*

.. image:: who.alcohol_consumption.png
   :height: 300px
   :width: 300 px
   :alt: Density estimation using alcohol consumption data from the WHO.
   :align: right

SUFTware (Statistics Using Field Theory) provides fast and lightweight Python
implementations of Bayesian Field Theory algorithms for low-dimensional
statistical inference. SUFTware currently supports the one-dimensional
density estimation algorithm DEFT, described in [#Chen2018]_,
[#Kinney2015]_, and [#Kinney2014]_. The image on the right shows DEFT applied
to alcohol consumption data from
the World Health Organization. This computation took about 0.25 seconds on
a standard laptop computer.

Code for this and other examples can be found on the :doc:`examples` page.
The :doc:`tutorial` page contains a short tutorial on how to use SUFTware.
The :doc:`documentation` page details the SUFTware API.

Installation
------------

SUFTware can be installed from
`PyPI <https://pypi.python.org/pypi/suftware>`_ using the pip package
manager (version 9.0.0 or higher). At the command line::

    pip install suftware

The code for SUFTware is open source and available on
`GitHub <https://github.com/jbkinney/suftware>`_.


Quick Start
-----------

To make the figure shown above, do this from within Python::

   import suftware as sw
   sw.demo()

Resources
---------

.. toctree::

    tutorial
    examples
    documentation

Contact
-------

For technical assistance or to report bugs, please
contact `Ammar Tareen <tareen@cshl.edu>`_.

For more general correspondence, please
contact `Justin Kinney <jkinney@cshl.edu>`_.

Other links:

- `Kinney Lab <http://kinneylab.labsites.cshl.edu/>`_
- `Simons Center for Quantitative Biology <https://www.cshl.edu/research/quantitative-biology/>`_
- `Cold Spring Harbor Laboratory <https://www.cshl.edu/>`_


References
----------

.. [#Chen2018] Chen W, Tareen A, Kinney JB (2018) `Density estimation on small datasets. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.160605>`_ *Phys Rev Lett 121:160605*.
   :download:`PDF <ChenEtAl2018PRL.pdf>`.

.. [#Kinney2015] Kinney JB (2015) `Unification of field theory and maximum entropy methods for learning probability densities. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.032107>`_ *Phys Rev E* 92:032107.
   :download:`PDF </Kinney2015.pdf>`.

.. [#Kinney2014] Kinney JB (2014) `Estimation of probability densities using scale-free field theories. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.011301>`_ *Phys Rev E* 90:011301(R).
   :download:`PDF <Kinney2014.pdf>`.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
