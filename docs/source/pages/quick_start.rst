============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/mafese />`_::

   $ pip install mafese==0.1.1


* Install directly from source code::

   $ git clone https://github.com/thieu1995/mafese.git
   $ cd mafese
   $ python setup.py install


===============
Lib's structure
===============

Current's structure::

   docs
   examples
   mafese
      wrapper
         recursive.py
         sequential.py
      filter.py
      utils
         correlation.py
         encoder.py
         estimator.py
         validator.py
      __init__.py
      selector.py
   README.md
   setup.py

=====
Usage
=====

After installation, you can import MAFESE as any other Python module::

   $ python
   >>> import mafese
   >>> mafese.__version__


Let's go through some examples.

========
Examples
========

First, you need to load your dataset, or you can load own available datasets::

	# Load available dataset from MAFESE
	from mafese import get_dataset

	# Try unknown data
	get_dataset("unknown")
	# Enter: 1

	data = get_dataset("Arrhythmia")


Load your own dataset if you want::

	import pandas as pd
	from mafese import Data

	# load X and y
	# NOTE mafese accepts numpy arrays only, hence the .values attribute
	dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
	X, y = dataset[:, 0:-1], dataset[:, -1]
	data = Data(X, y)


Next, split dataset into train and test set::

	data.split_train_test(test_size=0.2, inplace=True)
	print(data.X_train[:2].shape)
	print(data.y_train[:2].shape)


Next, how to use Recursive wrapper-based method::

	from mafese.wrapper.recursive import Recursive

	# define mafese feature selection method
	feat_selector = Recursive(problem="classification", estimator="rf", n_features=5)

	# find all relevant features - 5 features should be selected
	feat_selector.fit(data.X_train, data.y_train)

	# check selected features - True (or 1) is selected, False (or 0) is not selected
	print(feat_selector.selected_feature_masks)
	print(feat_selector.selected_feature_solution)

	# check the index of selected features
	print(feat_selector.selected_feature_indexes)

	# call transform() on X to filter it down to selected features
	X_train_selected = feat_selector.transform(data.X_train)
	X_test_selected = feat_selector.transform(data.X_test)


Or, how to use Sequential wrapper-based method::

	from mafese.wrapper.sequential import Sequential

	# define mafese feature selection method
	feat_selector = Sequential(problem="classification", estimator="knn", n_features=3, direction="forward")

	# find all relevant features - 5 features should be selected
	feat_selector.fit(data.X_train, data.y_train)

	# check selected features - True (or 1) is selected, False (or 0) is not selected
	print(feat_selector.selected_feature_masks)
	print(feat_selector.selected_feature_solution)

	# check the index of selected features
	print(feat_selector.selected_feature_indexes)

	# call transform() on X to filter it down to selected features
	X_train_selected = feat_selector.transform(data.X_train)
	X_test_selected = feat_selector.transform(data.X_test)


Or, how to use Filter-based feature selection with different correlation methods::

	from mafese.filter import Filter

	# define mafese feature selection method
	feat_selector = Filter(problem='classification', method='SPEARMAN', n_features=5)

	# find all relevant features - 5 features should be selected
	feat_selector.fit(data.X_train, data.y_train)

	# check selected features - True (or 1) is selected, False (or 0) is not selected
	print(feat_selector.selected_feature_masks)
	print(feat_selector.selected_feature_solution)

	# check the index of selected features
	print(feat_selector.selected_feature_indexes)

	# call transform() on X to filter it down to selected features
	X_train_selected = feat_selector.transform(data.X_train)
	X_test_selected = feat_selector.transform(data.X_test)

For more usage examples please look at [examples](/examples) folder.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
