.. MAFESE documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MAFESE's documentation!
==================================

.. image:: https://img.shields.io/badge/release-0.1.2-yellow.svg
   :target: https://github.com/thieu1995/mafese/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/mafese

.. image:: https://badge.fury.io/py/mafese.svg
   :target: https://badge.fury.io/py/mafese

.. image:: https://img.shields.io/pypi/pyversions/mafese.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/status/mafese.svg
   :target: https://img.shields.io/pypi/status/mafese.svg

.. image:: https://img.shields.io/pypi/dm/mafese.svg
   :target: https://img.shields.io/pypi/dm/mafese.svg

.. image:: https://github.com/thieu1995/mafese/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/mafese/actions/workflows/publish-package.yaml

.. image:: https://pepy.tech/badge/mafese
   :target: https://pepy.tech/project/mafese

.. image:: https://img.shields.io/github/release-date/thieu1995/mafese.svg
   :target: https://img.shields.io/github/release-date/thieu1995/mafese.svg

.. image:: https://readthedocs.org/projects/mafese/badge/?version=latest
   :target: https://mafese.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: http://isitmaintained.com/badge/resolution/thieu1995/mafese.svg
   :target: http://isitmaintained.com/project/thieu1995/mafese

.. image:: http://isitmaintained.com/badge/open/thieu1995/mafese.svg
   :target: http://isitmaintained.com/project/thieu1995/mafese

.. image:: https://img.shields.io/github/contributors/thieu1995/mafese.svg
   :target: https://img.shields.io/github/contributors/thieu1995/mafese.svg

.. image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. image:: https://zenodo.org/badge/545209353.svg
   :target: https://doi.org/10.5281/zenodo.7969042

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


MAFESE (Metaheuristic Algorithms for FEature SElection) is the largest python library focused on feature selection
using meta-heuristic algorithms.

* **Free software:** GNU General Public License (GPL) V3 license
* **Total Wrapper-based (Metaheuristic Algorithms)**: > 170 methods
* **Total Filter-based (Statistical-based)**: > 6 methods
* **Total classification dataset**: > 20 datasets
* **Total estimator methods**: > 3 methods
* **Total performance metrics (as fitness)**: > 10 metrics
* **Documentation:** https://mafese.readthedocs.io/en/latest/
* **Python versions:** 3.7.x, 3.8.x, 3.9.x, 3.10.x, 3.11.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, matplotlib, mealpy, permetrics


Features
--------

- Our library provides all state-of-the-art feature selection methods:
   + Filter-based FS
   + Embedded-based FS
   + Wrapper-based FS
      + Sequential-based: forward and backward
      + Recursive-based
      + MHA-based: Metaheuristic Algorithms
- We have implemented all feature selection methods based on scipy, scikit-learn and numpy to increase the speed of the algorithms.


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/mafese.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
