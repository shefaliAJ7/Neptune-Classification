import os
#from typing import List

CURRENT_DATA = "resources/"

CSV_DATA_PATH = CURRENT_DATA + 'raw_data_csvfiles/MergedFiles-SALLY-struck/'
PROCESSED_DATA_PATH = CURRENT_DATA + 'Modelling/'
RESULTS_DATA_PATH = CURRENT_DATA + 'Results/'
#Model_type for multi or binary
#MODEL_TYPE = 'BinarySentenceClassifier'
MODEL_TYPE = 'SentenceClassifier'
#epoch = 10 for binary, 40 for multi

#GLOVE_PATH = os.path.abspath(os.path.dirname(__file__))+'/../../glove.6B.100d.txt'
GLOVE_PATH = os.path.abspath(os.path.dirname(__file__))+'/../../GoogleNews-vectors-negative300.bin'
# Number between 0 and 1
TRAIN_TEST_SPLIT = 0.8
ep = 40

# hyper parameters
SENTENCE_LENGTH = 20
LSTM_SIZE = 20
PRED_CONFIDENCE = 0.4  # suggested 0.5

RAW_DATA='raw_data.h5'
CLEAN_DATA='clean_data.h5'
TRAIN_DATA='train_data.h5'
TEST_DATA='test_data.h5'

#Model_Name change according to model - Binary or Multi
MODEL_NAME = 'multiclass_model.h5'

#AGREEMENT=1 for Agree and AGREEMENt=0 for Disagree
AGREEMENT = 0

#TRAINING_LABEL , when using more than 1 coder and AGREEMENT=0, Then the first coder's sentences are considered
TRAINING_LABEL = ['label_SALLY', 'label_struck']
TRAINING_LABEL_DROP = ['label_SALLY','label_struck']
#TRAINING_LABEL = ['label_struck', 'label_SALLY']
#TRAINING_LABEL_DROP = ['label_struck', 'label_SALLY']
#TRAINING_LABEL = ['label_SALLY', 'label_Frenard']
#TRAINING_LABEL_DROP = ['label_SALLY','label_Frenard']
#TRAINING_LABEL = ['label_Frenard', 'label_SALLY']
#TRAINING_LABEL_DROP = ['label_Frenard', 'label_SALLY']
#TRAINING_LABEL = ['label_SALLY']
#TRAINING_LABEL_DROP = ['label_SALLY']
#TRAINING_LABEL = ['label_Frenard']
#TRAINING_LABEL_DROP = ['label_Frenard']
#TRAINING_LABEL = ['label_struck']
#TRAINING_LABEL_DROP = ['label_struck']
PREDICTION_FILE = 'Binary_Classification_test'
