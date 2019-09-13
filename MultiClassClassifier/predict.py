import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from Config import *

class predictClass:
    # hyper params, should use the same as used in train
    def __init__(self):
        self.sentence_length = SENTENCE_LENGTH
        # self.lstm_size = LSTM_SIZE
        self.confidence = PRED_CONFIDENCE  # suggested 0.5
        self.model = None
        self.tokenizer = None
        self.one_hot = None
        self.load_model()

    def load_model(self,text):
        # input a sentence or string
        # return a set of tags in the input
        # load model, tokenizer, and encoder
        self.model = load_model(PROCESSED_DATA_PATH+'model.h5')
        with open(PROCESSED_DATA_PATH+'tokenizer.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)

        with open(PROCESSED_DATA_PATH+'onehot.pickle', 'rb') as o:
            self.one_hot = pickle.load(o)

    def predict(self, text):
        text = [text]
        # encode and pad
        encoded_sent = self.tokenizer.texts_to_sequences(text)
        padded_sent = pad_sequences(encoded_sent, self.sentence_length, padding='post')

        # predict and decode to original tags(classes)
        preds = self.model.predict(padded_sent)
        y_classes = (preds > self.confidence).astype(int)
        tags = self.one_hot.inverse_transform(y_classes)
        return tags
