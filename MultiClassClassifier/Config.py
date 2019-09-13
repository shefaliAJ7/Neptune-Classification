PROCESSED_DATA_DIR = "ProcessedData/"
ANNOTATED_FILE_DIR = "CodingFiles/"
CURRENT_DATA = "InterCoderReliability/"

CSV_DATA_PATH = ANNOTATED_FILE_DIR + CURRENT_DATA
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR + CURRENT_DATA

GLOVE_PATH = '/Users/asudatascience/NeptuneProject/Repositories/Fall2019/glove.6B.100d.txt'
# hyper params, should use the same as used in train
SENTENCE_LENGTH = 20
LSTM_SIZE = 20
PRED_CONFIDENCE = 0.4  # suggested 0.5