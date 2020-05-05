# Classification Model - BERT

## Project Structure
### 1. Dataset: 
contains all the IRs in CSV format. BERT will take the 'text' column as X and 'label' column as Y.
### 2. BERT_binary.ipynb:
a google colab notebook with GPU, to apply binary classification using BERT. It classifies whether a sentence is U:useless or N: useful.
### 3. BERT_multiclass.ipynb: 
a google colab notebook with GPU, to apply multi-class classification using BERT. It classifies whether a sentence belongs to label: T, P, O, D, H
### 4. BERT_multiclass_kfold.ipunb: 
a google colab notebook with GPU, to apply K-Fold Cross Validation on multi-class classification using BERT. It calculates Test Accuracy for K=10 iterations with each iteration having a different set of Training and Test Data. It produces a line-graph that tells accuracies for each iteration.
### 5. Results: 
stores predictions and accuracies for Test Data for different Coder Types.
### 6. Results_KFold: 
stores Accuracy vs K-iteration graph-plots for different Coder Types.

