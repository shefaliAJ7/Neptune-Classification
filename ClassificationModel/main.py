#!/usr/bin/env python3.5
from Classifier import *
from Data import *
import shutil
import pandas as pd


# Multi Class Classifier
class MCC:
    def __init__(self):
        self.create_dataset = CreateDataset()
        self.train = Train(force_retrain=True)
        self.predict = Predict()
        self.train_accuracy = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.result_df = None
        self.final_result = pd.DataFrame(columns = ['Coder','Train_Accuracy','Val_Accuracy','Test_Accuracy','#TestData','#AllData'])

    def train_model(self):
        self.train_accuracy, self.val_accuracy = self.train.train_model()

    def predict_labels(self):
        self.predict.load_model()
        self.predict.predict_data()
        self.result_df, self.test_accuracy = self.predict.analyze()
        if not os.path.exists(RESULTS_DATA_PATH):
            os.mkdir(RESULTS_DATA_PATH)
        self.result_df.to_csv(RESULTS_DATA_PATH + PREDICTION_FILE + ".csv", header=True)

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
        print(self.final_result)
        filename = RESULTS_DATA_PATH+"Final_Results.csv"
        if not os.path.isfile(filename):
            self.final_result.to_csv(filename, header=True)
        else:
            self.final_result.to_csv(filename, mode='a', header=False)


if __name__ == '__main__':
    shutil.rmtree(PROCESSED_DATA_PATH)
    mcc = MCC()
    mcc.train_model()
    mcc.predict_labels()
    mcc.get_results()
