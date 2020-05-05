"""Configurations for implementing Classification

Provides configurations for File Path, Type of Classifier, Model Configurations
and labels by coders.

The Classification will be done for different Coder Types:
1 . Sally - for all sentences labelled by sally - [label_SALLY]
2. Trixy - for all sentences labelled by Trixy - [label_struck]
3. Frenard - for all sentences labelled by Frenard - [label_Frenard]
4. Sally_Trixy_Agree - for all sentences labelled by Sally and Trixy where they gave same labels - 
	TRAINING_LABEL = [label_SALLY, label_struck] with AGREEMENT = 1
5. Sally_Frenard_Agree - for all sentences labelled by Sally and Trixy where they gave same labels - 
	TRAINING_LABEL = [label_SALLY, label_Frenard] with AGREEMENT = 1
6. Sally_Trixy_DisAgree - for all sentences labelled by Sally where Sally and Trixy gave different labels - 
	TRAINING_LABEL = [label_SALLY, label_struck] with AGREEMENT = 0
7. Trixy_Sally_DisAgree - for all sentences labelled by Trixy where Sally and Trixy gave different labels - 
	TRAINING_LABEL = [label_struck, label_SALLY] with AGREEMENT = 0
8. Sally_Frenard_DisAgree - for all sentences labelled by Sally where Sally and Frenard gave different labels - 
	TRAINING_LABEL = [label_SALLY, label_Frenard] with AGREEMENT = 0
9. Frenard_Sally_DisAgree - for all sentences labelled by Frenard where Sally and Frenard gave different labels - 
	TRAINING_LABEL = [label_Frenard, label_SALLY] with AGREEMENT = 0
"""
import os
from typing import List

# Folder which contains all the resouce files
CURRENT_DATA = "resources/"

# Path which has the Dataset, Change according to coder-type for which classification needs to be done
CSV_DATA_PATH = CURRENT_DATA + 'raw_data_csvfiles/MergedFiles-SALLY-Frenard/'

# Path where the Model and its checkpoints will be saved
PROCESSED_DATA_PATH = CURRENT_DATA + 'Modelling/'

# Path where the Prediction Results and Accuracy Results will be saved
RESULTS_DATA_PATH = CURRENT_DATA + 'Results/'

# Specifies the Type of Classification Model which needs to be created
#MODEL_TYPE = 'BinarySentenceClassifier'
MODEL_TYPE = 'SentenceClassifier'

# Epochs are different for different Model Types
#epoch = 10 for binary labels, 40 for multi-labels
ep = 40

# File needed to perform Word2Vec - Pre-Trained W2V - GloVe
GLOVE_PATH = os.path.abspath(os.path.dirname(__file__))+'/../../glove.6B.100d.txt'

TRAIN_TEST_SPLIT = 0.8

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

#AGREEMENT=1 for Agree and AGREEMENt=0 for Disagree between coders
AGREEMENT = 1

# Coder Types - Specifies labels labelled by coders
TRAINING_LABEL = ['label_SALLY', 'label_Frenard']

# After Data Cleaning, we will get one 'label' needed to perform classification, these labels will be dropped
TRAINING_LABEL_DROP = ['label_SALLY','label_Frenard']

# Specifies filename for file which will store prediction results of Test Data
PREDICTION_FILE = 'Sally_Frenard_Agree'