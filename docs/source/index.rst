.. MAFESE documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MAFESE's documentation!
==================================

.. image:: https://img.shields.io/badge/release-1.0.0-yellow.svg
   :target: https://github.com/thieu1995/mafese/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/mafese

.. image:: https://badge.fury.io/py/mafese.svg
   :target: https://badge.fury.io/py/mafese

.. image:: https://img.shields.io/pypi/pyversions/mafese.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/dm/mafese.svg
   :target: https://img.shields.io/pypi/dm/mafese.svg

.. image:: https://static.pepy.tech/badge/mafese
   :target: https://pepy.tech/project/mafese

.. image:: https://github.com/thieu1995/mafese/actions/workflows/test.yml/badge.svg
   :target: https://github.com/thieu1995/mafese/actions/workflows/test.yml

.. image:: https://readthedocs.org/projects/mafese/badge/?version=latest
   :target: https://mafese.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/badge/DOI-10.1016%2Fj.future.2024.06.006-blue
   :target: https://doi.org/10.1016/j.future.2024.06.006

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


MAFESE (Metaheuristic Algorithms for FEature SElection) is the largest python library focused on feature selection
using meta-heuristic algorithms.

* **Free software:** GNU General Public License (GPL) V3 license
* **Total Wrapper-based (Metaheuristic Algorithms)**: > 200 methods
* **Total Filter-based (Statistical-based)**: > 15 methods
* **Total Embedded-based (Tree and Lasso)**: > 10 methods
* **Total Unsupervised-based**: >= 4 methods
* **Total datasets**: >= 30 (47 classifications and 7 regressions)
* **Total performance metrics**: >= 61 (45 regressions and 16 classifications)
* **Total objective functions (as fitness functions)**: >= 61 (45 regressions and 16 classifications)
* **Documentation:** https://mafese.readthedocs.io/en/latest/
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, plotly, kaleido


Features
--------

- Our library provides all state-of-the-art feature selection methods:
   + Filter-based FS
   + Embedded-based FS
      + Regularization (Lasso-based)
      + Tree-based methods
   + Wrapper-based FS
      + Sequential-based: forward and backward
      + Recursive-based
      + MHA-based: Metaheuristic Algorithms
   + Unsupervised-based FS
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
