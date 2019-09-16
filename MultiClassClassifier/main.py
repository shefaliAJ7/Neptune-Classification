#!/usr/bin/env python3.5
from MultiClassClassifier import *
from Data import *
import shutil


# Multi Class Classifier
class MCC:
    def __init__(self):
        self.create_dataset = CreateDataset()
        self.train = Train(force_retrain=False)
        self.predict = Predict()

    def train_model(self):
        self.train.train_model()

    def predict_labels(self):
        self.predict.load_model()
        self.predict.predict_data()
        self.predict.analyze()


if __name__ == '__main__':
    shutil.rmtree(PROCESSED_DATA_PATH)
    mcc = MCC()
    mcc.train_model()
    mcc.predict_labels()
