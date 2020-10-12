.. -*- mode: rst -*-

|Python36|_ |Travis|_ |Codecov|_


.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/scikit-learn

.. |Travis| image:: https://travis-ci.com/hcherkaoui/carpet.svg?branch=master
.. _Travis: https://travis-ci.com/hcherkaoui/carpet

.. |Codecov| image:: https://codecov.io/gh/hcherkaoui/carpet/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hcherkaoui/carpet


Carpet: Neural Net based solver for the 1d-TV problem
=====================================================

An adaptive optimization package for the 1d-TV problem for research purpose, proposed in the following paper:  

`Hamza Cherkaoui`_, `Jeremias Sulam`_ and `Thomas Moreau`_ "`Learning to solve TV regularised problems with unrolled algorithms`_", accepted at NeurIPS 2020.


This package implements:

- **Classical solver for TV-regularized optimization problems**: (primal-dual) ISTA, (primal-dual) FISTA, Condat-Vu.

- **Learnable algorithms**: All the iterative versions cited above.

.. Links to different projects


.. _pytorch: https://pytorch.org/
.. _tensorflow: https://tensorflow.org/
.. _numpy: https://numpy.org/
.. _prox_tv: https://github.com/albarji/proxTV/


Important links
===============

- Official source code repo: https://github.com/hcherkaoui/carpet

Dependencies
============

The required dependencies to use the software are:

* Numpy >= 1.14.0
* Scipy >= 1.0.0
* Joblib >= 0.11
* Torch >= 1.4.0
* Matplotlib >= 2.1.2
* Prox_tv

License
=======

All material is Free Software: BSD license (3 clause).

Installation
============

In order to perform the installation, run the following command from the carpet directory::

    python3 setup.py install --user

To run all the tests, run the following command from the carpet directory::

    pytest

Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/hcherkaoui/carpet

or if you have write privileges::

    git clone git@github.com:hcherkaoui/carpet
    
    
.. _`Thomas Moreau`: https://tommoral.github.io
.. _`Hamza Cherkaoui`: https://hcherkaoui.github.io/
.. _`Jeremias Sulam`: https://sites.google.com/view/jsulam
.. _`Learning to solve TV regularised problems with unrolled algorithms`: https://hal.archives-ouvertes.fr/hal-02954181
