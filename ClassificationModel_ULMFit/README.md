# Classification Model - ULMFit

## Project Structure
### 1. Dataset: 
contains all the IRs in CSV format. ULMFit will take the 'text' column as X and 'label' column as Y.
### 2. ULMFit.ipynb:
a google colab notebook with GPU, to apply binary classification and  multi-class classification using BERT. Binary Classification classifies whether a sentence is U:useless or N: useful. Multi-class classification classifies whether a sentence belongs to label: T, P, O, D, H
### 4. ULMFit_multiclass_kfold.ipunb: 
a google colab notebook with GPU, to apply K-Fold Cross Validation on multi-class classification using ULMFit. It calculates Test Accuracy for K=10 iterations with each iteration having a different set of Training and Test Data. It produces a line-graph that tells accuracies for each iteration.
### 5. Results: 
stores predictions and accuracies for Test Data for different Coder Types.
### 6. Results_KFold: 
stores Accuracy vs K-iteration graph-plots for different Coder Types.
