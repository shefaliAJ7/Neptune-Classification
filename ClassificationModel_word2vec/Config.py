import os
from typing import List

CURRENT_DATA = "resources/"

CSV_DATA_PATH = CURRENT_DATA + 'raw_data_csvfiles/For_binary_classification/'
PROCESSED_DATA_PATH = CURRENT_DATA + 'Modelling/'
RESULTS_DATA_PATH = CURRENT_DATA + 'Results/'
MODEL_TYPE = 'BinarySentenceClassifier'

#GLOVE_PATH = os.path.abspath(os.path.dirname(__file__))+'/../../glove.6B.100d.txt'
GLOVE_PATH = os.path.abspath(os.path.dirname(__file__))+'/../../GoogleNews-vectors-negative300.bin'
# Number between 0 and 1
TRAIN_TEST_SPLIT = 0.8

# hyper parameters
SENTENCE_LENGTH = 20
LSTM_SIZE = 20
PRED_CONFIDENCE = 0.4  # suggested 0.5

RAW_DATA='raw_data.h5'
CLEAN_DATA='clean_data.h5'
TRAIN_DATA='train_data.h5'
TEST_DATA='test_data.h5'

MODEL_NAME = 'binaryclass_model.h5'
AGREEMENT = 0
#TRAINING_LABEL = ['label_struck','label_SALLY']
#TRAINING_LABEL_DROP = ['label_SALLY','label_Frenard','label_struck']
TRAINING_LABEL = ['label_SALLY']
TRAINING_LABEL_DROP = ['label_SALLY']
#TRAINING_LABEL = ['label_Frenard']
#TRAINING_LABEL_DROP = ['label_Frenard']
#TRAINING_LABEL = ['label_struck']
#TRAINING_LABEL_DROP = ['label_struck']
PREDICTION_FILE = 'Binary_Classifcation_Result(310)'
