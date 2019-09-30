import os
# Retrain model
RETRAIN = True

# Hyper parameters
SENTENCE_LENGTH = 20
LSTM_SIZE = 20
# Model Params
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 16
EPOCHS = 40
VERBOSE = 1 # Display output during training
# Number between 0 and 1
TRAIN_TEST_SPLIT = 0.6
PRED_CONFIDENCE = 0.5  # Confidence score for multi-class sentence classifier

# Choose Classifier
MODEL_TYPE='BinarySentenceClassifier'
# MODEL_TYPE='SentenceClassifier'
# MODEL_TYPE='MultiClassSentenceClassifier'

#Choose class label from data
TRAINING_LABEL = 'class'
# TRAINING_LABEL = 'label_SALLY'
# TRAINING_LABEL = 'label_Frenard'
# TRAINING_LABEL = 'label_struck'

#Resulting Prediction file name
PREDICTION_FILE = 'Result.csv'

# Relative Path handling
PROCESSED_DATA_DIR = "ProcessedData/"
ANNOTATED_FILE_DIR = "CodingFiles/"
# CURRENT_DATA = "InterCoderReliability/"
CURRENT_DATA = 'Sally/'

CSV_DATA_PATH = ANNOTATED_FILE_DIR + CURRENT_DATA
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR + CURRENT_DATA

GLOVE_PATH = os.path.abspath(os.path.dirname(__file__))+'/../../glove.6B.100d.txt'

if MODEL_TYPE == 'BinarySentenceClassifier':
    MODEL_NAME = 'binary_sentence_classifier_model.h5'
elif MODEL_TYPE == 'SentenceClassifier':
    MODEL_NAME = 'sentence_classifier_model.h5'
elif MODEL_TYPE == 'MultiClassSentenceClassifier':
    MODEL_NAME = 'multi_class_sentence_classifier_model.h5'
MODEL_PATH = PROCESSED_DATA_PATH+MODEL_TYPE+'/'

# Processed data file names
RAW_DATA='raw_data.h5'
CLEAN_DATA='clean_data.h5'
TRAIN_DATA='train_data.h5'
TEST_DATA='test_data.h5'
