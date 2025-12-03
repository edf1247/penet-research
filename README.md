# penet-research
Repo for penet research

## Usage
1. Run the PENet model on the validation and test sets
2. Run the text classifier
3. Pass the validation and test predictions for both models to the fusion model

## PENet Model
Model from [this repo](https://github.com/marshuang80/penet)

Added Optuna integration for hyperparameter optimization, however this resulted in a marginal increase in AUROC, ~.05%. The final implementation utilizes predictions from the base model.

## Text classification

Implemented XGBoost with early stopping

## Combining the Two

We implemented ensemble stacking:
- The image and csv data came pre-split into training, validation, and test sets
- Train the PENet model and the text classifier on the training data alone
    - Create validation sets for the model from the training data, holding the annotated validation set out 
- Test the models on the validation set, saving the probabilities
- Train a logistic regression model on those probabilities
- Test the models on the test set, saving the probabilities
- Test the ensemble on those probabilities


## Results

| AUROC | Accuracy(default classification threshold) | Accuracy(theshold of 0.440) |
|---|---|---|
| 0.968 | 0.9012345679012346 | 0.9135802469135802 |

The PENet model had an AUROC of 0.84 on the test set, while our ensemble has an AUROC of 0.968. This is a nearly 13 point increase in the models abilitiy to rank positive cases higher than negative cases. Also, the model's accuracy of 91% makes it easily applicable to a clinical setting. 
