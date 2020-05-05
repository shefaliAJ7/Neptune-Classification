# Classification Model - GloVe + CNN + LSTM

## Project Structure

### 1. resources:
- Modelling - contains model-weight files, their checkpoints, cleaned .h5 training and test files, etc.
- raw_data_csv_files - contains all the IRs in CSV format. Model will take the 'text' column as X and 'label' column as Y.
- Results - stores predictions and accuracies of Test Data of all 'Dataset Types'.
- Results_KFold - stores Accuracy vs K-iteration graph-plots of all 'Dataset Types'.

### 2. main.py:
This file provides the "main" function and is the starting point to run the classification model (binary or multi-class) for different 'Dataset Types'. It creates the Classification model, cleans and processes the dataset, trains the data on it, predicts the labels and saves the result. All the results are saved separately and also combined in one csv file. We can run the Classification Model or Perform K-Fold Cross Validation of this Classification Model.

### 3. Classifier.py:
This file creates a Classification Model, use it to train on the Training Data and use it to predict the labels of Test data.
The classification model is a combination of GloVe + CNN + LSTM. It generates models for Binary Classification and Multi-Class Classification.
This File also provides K-Fold Cross Validation to test the biases in the Classification Model.
Three types of Classification Model can be generated:
- Sentence Classifier - gets one label out of (T, P, O, D, H) labels
- Multi-class sentence Classifier - gets multiple label out of (T, P, O, D, H) labels
- Binary Classifier - gets one label out of (U, N) two labels

### 4. Data.py:
This file is used for collecting the Dataset that contains the sentences labelled
by different coders. <br>
It helps to create different 'Dataset Types' depending upon 'Coder Types'.
1. Binary Classification: Equal no. of U: useful and N: useless sentences labelled by Sally are taken
2. Sentence Classification or Multi-Class sentence Classification
    - 1 Coder
    - 2 Coders
        - Similar labelled useful sentences by the two coders
        - Different labelled useful sentences  by the two coders
    - 3 Coders
        - Similar labelled useful sentences by the three coders
    <br>
The data is then divided into Training and Test data and saved as a dataframe.

### 4. Config.py: 
Provides configurations for File Path, Type of Classifier, Model hyperparameters and Dataset Types.
