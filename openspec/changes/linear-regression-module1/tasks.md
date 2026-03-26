# Implementation Tasks: Linear Regression

- [ ] **Task 1: Model the "From Scratch" Engine**
  - File: `module_1_foundations/01_linear_regression/scratch.py`
  - Write `LinearRegressionFromScratch` mapping out the `fit` and `predict` routines using pure NumPy.
  - **Dependencies:** None

- [ ] **Task 2: Model the "Framework" Engine**
  - File: `module_1_foundations/01_linear_regression/framework.py`
  - Implement a simple wrapper returning a fitted `sklearn.linear_model.LinearRegression` line of best fit on X and y inputs.
  - **Dependencies:** None

- [ ] **Task 3: Assemble the interactive Notebook**
  - File: `module_1_foundations/01_linear_regression/tutorial.ipynb`
  - Fetch California Housing `MedInc` dataset.
  - Run both models on the test data.
  - Export dual `matplotlib` scatter plots verifying the lines match.
  - **Dependencies:** Task 1, Task 2

- [ ] **Task 4: Finalize Module Documentation**
  - File: `module_1_foundations/01_linear_regression/README.md`
  - Write the mathematical formulas for Mean Squared Error and Gradient Descent. Explain simply how weights update themselves to minimize the loss.
  - **Dependencies:** Task 3
