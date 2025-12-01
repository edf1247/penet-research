# penet-research
Repo for penet research

## Usage
1. Run the PENet model on the validation and test sets
2. Run the text classifier on the validation and test sets
3. Pass the validation and test predictions for both models to the fusion model

## PENet Model
Model from [this repo](https://github.com/marshuang80/penet)

Added Optuna integration for hyperparameter optimization. 

## Text classification

Implemented a simple logistic regression model for classification using patient data.


## Combining the Two

Instead of taking a weighted sum, we took the following approach:
- Train the PeNet model and the text classifier on the training data alone (set a validation set and test set aside)
- Get the model predictions for the validation set, and train a logistic regression model on those predicted probabilities
- Then, have the PeNET model and text classifier make predictions on the test data. 
- Feed those predictions into the ensemble, having it make predictions based on that data.


## Results

The ensemble reported an AUROC of 95.8% on the test set, compared to the 84% AUROC of the PENet model. This is over a 14% increase in model performance.

With the new text classifier using xgboost, we get these results:

AUC-ROC: 0.968
Accuracy with default threshold: 0.8827160493827161
Accuracy with optimal threshold (0.230): 0.9197530864197531

Notably, the accuracy with the optimal threshold is significantly higher than with the previous regressor.