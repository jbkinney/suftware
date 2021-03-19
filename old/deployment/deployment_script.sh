#!/bin/bash
sudo python setup.py sdist bdist_wheel
virtualenv your-package-sdist
source your-package-sdist/bin/activate
your-package-sdist/bin/pip install dist/suftware-0.1.tar.gz
cd your-package-sdist
bin/python
echo "Ensure package installed correctly by importing it and running a basic test_for_mistake"
echo " If test_for_mistake successful, run: twine upload pypi dist/suftware-0.1*"