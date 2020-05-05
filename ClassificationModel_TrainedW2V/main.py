"""Start the Classifier from here

This file provides the "main" function and is the starting point to run the 
classification model for different coders.

It creates the Classification model, trains the data on it, predicts the labels
and saves the result. All the results are saved seperately and also combined in
one csv file.

We can run the Classification Model or Perform K-Fold Cross Validation of this
Classification Model.
"""

#!/usr/bin/env python3.5
from Classifier import *
from Data import *
import shutil
import pandas as pd
import logging

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.WARNING)
#
# def set_tf_loglevel(level):
#     if level >= logging.FATAL:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     if level >= logging.ERROR:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#     if level >= logging.WARNING:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#     else:
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#     logging.getLogger('tensorflow').setLevel(level)
#
# set_tf_loglevel(logging.INFO)


#!/usr/bin/env python3.5
from Classifier import *
from Data import *
import shutil
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Multi Class Classifier
class MCC:
    def __init__(self):
        self.create_dataset = CreateDataset()
        self.train = Train(force_retrain=True)
        self.predict = Predict()
        self.train_accuracy = None
        self.val_accuracy = None
        self.test_accuracy = None
        self.result_df = None
        self.labels = None
        self.conf_mat = None
        self.precision = None
        self.recall = None
        self.f1score = None
        self.final_result = pd.DataFrame(columns = ['Coder','Train_Accuracy','Val_Accuracy','Test_Accuracy','#TestData','#AllData'])

    # Train model on Trainning data or do K-Fold Cross Validation on Training Data
    def train_model(self, is_it_k_fold):
        if is_it_k_fold:
            self.train.train_model_with_kfold()
        else:
            self.train_accuracy, self.val_accuracy = self.train.train_model()

    # Predict labels of Test Data
    def predict_labels(self):
        self.predict.load_model()
        self.predict.predict_data()
        self.result_df, self.test_accuracy = self.predict.analyze()
        self.labels = list(set(self.result_df['Y']))
        self.conf_mat = confusion_matrix(self.result_df['Y'], self.result_df['Predicted_Values'], self.labels)

        if MODEL_TYPE == 'BinarySentenceClassifier':
            self.precision = precision_score(self.result_df['Y'], self.result_df['Predicted_Values'], pos_label = 'N')
            self.recall = recall_score(self.result_df['Y'], self.result_df['Predicted_Values'], pos_label='N')
            self.f1score = f1_score(self.result_df['Y'], self.result_df['Predicted_Values'], pos_label='N')

        if not os.path.exists(RESULTS_DATA_PATH):
            os.mkdir(RESULTS_DATA_PATH)
        self.result_df.to_csv(RESULTS_DATA_PATH + PREDICTION_FILE + ".csv", header=True)

    # Save Test Data with its predicted labels - seperately, or append in Final_Result.csv
    def get_results(self):
        list = []
        list.append(PREDICTION_FILE)
        self.final_result['Coder'] = list
        list = []
        list.append(self.train_accuracy)
        self.final_result['Train_Accuracy'] = list
        list = []
        list.append(self.val_accuracy)
        self.final_result['Val_Accuracy'] = list
        list = []
        list.append(self.test_accuracy)
        self.final_result['Test_Accuracy'] = list
        list = []
        list.append(len(self.result_df['text']))
        self.final_result['#TestData'] = list
        list = []
        list.append(self.create_dataset.total_data)
        self.final_result['#AllData'] = list
        list = []
        list.append(self.labels)
        self.final_result['Classes'] = list
        list = []
        list.append(self.conf_mat)
        self.final_result['Confusion Matrix'] = list

        if MODEL_TYPE == 'BinarySentenceClassifier':
            list = []
            list.append(self.precision)
            self.final_result['Precision'] = list
            list = []
            list.append(self.recall)
            self.final_result['Recall'] = list
            list = []
            list.append(self.f1score)
            self.final_result['F1'] = list

        print(self.final_result)
        filename = RESULTS_DATA_PATH+"Final_Results.csv"
        if not os.path.isfile(filename):
            self.final_result.to_csv(filename, header=True)
        else:
            self.final_result.to_csv(filename, mode='a', header=False)

# Main Function to start the Classification Process
if __name__ == '__main__':
    doing_k_fold_validation = True
    shutil.rmtree(PROCESSED_DATA_PATH)
    mcc = MCC()
    mcc.train_model(doing_k_fold_validation)
    if doing_k_fold_validation == False:
        mcc.predict_labels()
        mcc.get_results()
