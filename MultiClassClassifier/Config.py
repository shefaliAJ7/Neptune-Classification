PROCESSED_DATA_DIR = "ProcessedData/"
ANNOTATED_FILE_DIR = "CodingFiles/"
CURRENT_DATA = "InterCoderReliability/"

CSV_DATA_PATH = ANNOTATED_FILE_DIR + CURRENT_DATA
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR + CURRENT_DATA

GLOVE_PATH = '/Users/asudatascience/NeptuneProject/Repositories/Fall2019/glove.6B.100d.txt'


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

MODEL_NAME = 'multiclass_model.h5'
