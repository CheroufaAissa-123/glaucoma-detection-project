# Ahcene - Data Preparation and Model Evaluation

## Scope

This contribution area covers the parts of the project related to preparing the data pipeline for training and evaluating glaucoma classification models.

## Main Responsibilities

- Define and maintain image preprocessing choices used before training
- Configure data augmentation pipelines adapted to retinal fundus images
- Prepare train, validation, and test splits consistently
- Handle class imbalance during training data loading
- Track and interpret evaluation metrics for the trained model

## Current Project Mapping

The current implementation for this scope is mainly located in `notebooks/code_principale.ipynb`.

Relevant sections already present in the notebook:

- Transformations and preprocessing with Albumentations
- CLAHE-based contrast enhancement
- Dataset loading with `ImageFolder`
- Train/validation/test preparation
- Weighted sampling for imbalance handling
- Evaluation with accuracy, precision, recall, F1-score, AUC, and confusion matrix

## Expected Deliverables For This Branch

- Clear preprocessing and augmentation configuration
- Reproducible dataset preparation steps
- Documented evaluation procedure
- Clean reporting of the main validation and test metrics

## Notes

This branch is intended for work related to data preparation and evaluation only.
Inference API and frontend demo code remain outside the main scope of this contribution.
