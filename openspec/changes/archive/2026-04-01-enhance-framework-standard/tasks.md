# Tasks: Enhance Framework Standard (TinyGrad + PyTorch)

## Standards Implementation

- [x] **Update `.clinerules`**: Standardize the "TinyGrad-First" approach (Priority: TinyGrad > PyTorch > Scikit-Learn fallback).
- [x] **Update `spec.md`**: Formalize the new library dependencies and implementation logic in `openspec/specs/coding-standard/spec.md`.

## Module 1 Refactoring (01-05)

- [x] **01_linear_regression**: Implement `LinearRegressionFramework` using TinyGrad and PyTorch. Update `README.md` and `tutorial.py`.
- [x] **02_logistic_regression**: Implement `LogisticRegressionFramework` using TinyGrad and PyTorch. Update `README.md` and `tutorial.py`.
- [x] **03_knn**: (Discrete) No change needed (keeping `scikit-learn` or pure distance logic).
- [x] **04_naive_bayes**: (Discrete) No change needed (keeping `scikit-learn`).
- [x] **05_svm**: Implement `SVMFramework` using TinyGrad and PyTorch. Update `README.md` and `tutorial.py`.

## Global Verification

- [x] Run all `tutorial.py` scripts from Module 1 to verify the new dual-framework logic works correctly.
- [x] Ensure consistent naming conventions are followed in the updated files.
