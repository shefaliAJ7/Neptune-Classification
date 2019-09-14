#!/usr/bin/env python3.5
from MultiClassClassifier import *
from Data import *
from os import sys, path


# Multi Class Classifier
class MCC:
    def __init__(self):
        self.create_dataset = CreateDataset()
        self.train = Train()
        self.predict = Predict()

    def train(self):
        self.train.train_model()
        self.train.plot_history()

    def predict(self):
        self.predict.load_model()
        self.predict.predict()

if __name__ == '__main__':
    mcc = MCC()
    mcc.train()
    mcc.predict()
