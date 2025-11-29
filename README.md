# penet-research
Repo for penet research

## PeNet Model
Model from [this repo](https://github.com/marshuang80/penet)

Added Optuna integration for hyperparameter optimization. 
ToDo:
- compare optimized performance to penet_best
- Add XGBoost integration

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
