# Version 0.1.8

+ Fix bug load data from library.

---------------------------------------------------------------------

# Version 0.1.7

+ Remove some unknown datasets
+ Fix bug name in Kendall and Spearman functions of FilterSelector 
+ Add Relief-based family to FilterSelector 
  + Relief Original 
  + Relief-F 
  + VLS-Relief-F: Very Large Scale ReliefF 
+ Remove rtf transfer function in MhaSelector
+ Update save results file of MultiMhaSelector's evaluate() function.
+ Update examples for some Selector class

---------------------------------------------------------------------

# Version 0.1.6

+ Rename some public functions to private functions
+ Add MultiMhaSelector class
+ Add Matplotlib library and support visualization for MultiMhaSelector class
+ Add dependency plotly>=5.10.0 and kaleido >=0.2.1
+ Update examples for some Selector class
+ Replace evaluator module by evaluate method in Selector class.

---------------------------------------------------------------------

# Version 0.1.5

+ Add more regression and classification datasets
+ Update documents, examples, test
+ Remove matplotlib dependency

---------------------------------------------------------------------

# Version 0.1.4

+ Add Unsupervised-based methods:
  - "VAR": Variance Threshold method
  - "MAD": Mean Absolute Difference
  - "DR": Dispersion Ratio
  - "MCL": Multicollinearity method based on Variance Inflation Factor (VIF) value
+ Update documents, examples, test
+ Remove matplotlib dependency

---------------------------------------------------------------------


# Version 0.1.3

+ Relocate regression and classification datasets
+ Add Embedded Feature Selection:
  + Regularization methods (lasso-based)
  + Tree-based methods
+ Update documents, examples, test

---------------------------------------------------------------------


# Version 0.1.2

+ Add transfer utility
+ Add mealpy_util wrapper
+ Add MhaSelector class that holds all Metaheuristic Algorithm for Feature Selector
+ Add examples, tests, docs for MhaSelector class 
+ Update Data class 

---------------------------------------------------------------------

# Version 0.1.1

+ Add Github Action
+ Add citation
+ Add tests
+ Add docs
+ Rename classes:
  + Sequential to SequentialSelector
  + Recursive to RecursiveSelector
  + Filter to FilterSelector

---------------------------------------------------------------------

# Version 0.1.0 (First version)

+ Add project, Selector class, utils module.
+ Add wrapper module:
  + sequential-based
  + recursive-based
+ Add filter-based module.
+ Add examples
+ Add infors (ChangeLog.md, MANIFEST.in, LICENSE, README.md, requirements.txt)

