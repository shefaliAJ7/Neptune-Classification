#!/usr/bin/env python3.5
from Config import *
from Classifiers import *
from Data import *
import shutil


# Class used for creating objects of the Train and Predict class for further processing
class Classification:
    def __init__(self, force_retrain=True):
        self.create_dataset = CreateDataset()
        self.train = Train(force_retrain)
        self.predict = Predict()

    def train_model(self):
        self.train.train_model()

    def predict_labels(self):
        self.predict.load_model()
        self.predict.predict_data()
        self.predict.analyze()


if __name__ == '__main__':
    # This deletes folder corresponding to model in case RETRAIN is set to True
    # If you want to split the training and testing data differently, manually delete the corresponding files
    # present in the corresponding processed data folder
    if RETRAIN:
        try:
            shutil.rmtree(MODEL_PATH)
        except:
            # If folder does not exist
            pass
    classifier = Classification(force_retrain=RETRAIN)
    # Training the model
    classifier.train_model()
    # Predicting labels using model
    classifier.predict_labels()
