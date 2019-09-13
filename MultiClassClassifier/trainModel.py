from Config import *
import pandas as pd
import pickle
import numpy as np
# import re
# from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import string

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Embedding, Dropout, MaxPooling1D, LSTM
# from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
# from keras.models import load_model


class trainModel:
    def __init__(self):
        # load data
        # assumes input has first col string and second col set obj with classes
        self.df = pd.read_hdf(PROCESSED_DATA_PATH+'raw_data.h5', key='data')
        pd.set_option('display.expand_frame_repr', False)

        # encode classes
        self.one_hot = MultiLabelBinarizer()
        self.Y = self.one_hot.fit_transform(self.df['label_SALLY'])
        self.save_encoder()

        # hyperparameters
        self.vocabulary_size = 0 # set by self.build_tokenizer
        self.sentence_length = SENTENCE_LENGTH
        self.lstm_size = LSTM_SIZE
        self.number_of_classes = len(self.one_hot.classes_)

        # build tokenizer
        self.tokenizer = Tokenizer()
        self.build_tokenizer()

        # create embedding matrix
        self.embedding_matrix = self.create_embedding_matrix()

        # pad sentences
        self.padded_sent = self.pad_sentences()

        # create deep learning model
        self.model = self.generate_model()
        self.history = None

        # # train_model
        # self.train_model()


    def build_tokenizer(self):
        self.tokenizer.fit_on_texts(df['text'])
        self.vocabulary_size = len(self.tokenizer.word_index) + 1
        with open(PROCESSED_DATA_PATH+'tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # model structure
    def generate_model(self):
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, 100, input_length=self.sentence_length, weights=[self.embedding_matrix],
                            trainable=True))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(self.lstm_size))
        model.add(Dense(self.number_of_classes, activation='sigmoid'))
        # atleast one class match
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def save_encoder(self):
        # save encoder obj, can be used for prediction
        with open(PROCESSED_DATA_PATH+'onehot.pickle', 'wb') as handle:
            pickle.dump(self.one_hot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #embedding matrix
    def create_embedding_matrix(self):
        embeddings_index = dict()
        f = open('glove.6B.100d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        embedding_matrix = np.zeros((self.vocabulary_size, 100))
        for word, index in self.tokenizer.word_index.items():
            if index > self.vocabulary_size - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        return embedding_matrix

    # pad sentences
    def pad_sentences(self):
        encoded_sent = self.tokenizer.texts_to_sequences(self.df['text'])
        return pad_sequences(encoded_sent, self.sentence_length, padding='post')

    def train_model(self):
        self.history = self.model.fit(self.padded_sent, self.Y, validation_split=0.1, batch_size=512, epochs=10, verbose =1)
        self.model.save(PROCESSED_DATA_PATH+'model.h5')
        self.plot_history()

    # works only for accuracy not with categorical_accuracy
    def plot_history(self):
        history = self.history
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


