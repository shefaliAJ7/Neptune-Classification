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
from keras.models import load_model


# This class handles all the training portion of classifiers using neural networks
class Train(object):
    # The init function performs all the tasks in a sequential manner to perform training
    def __init__(self,training_data=TRAIN_DATA, force_retrain=False):
        # Check if model path exists and create it if it doesn't
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        else:
            # Check if model file exists and no RETRAIN is set to false to skip training
            if os.path.exists(MODEL_PATH + MODEL_NAME) and not force_retrain:
                self.ignore = True
                print("Skipping Training...")
                return
        self.ignore = False

        # load training data
        self.df = pd.read_hdf(PROCESSED_DATA_PATH+training_data, key='train')
        self.class_label_handler()
        self.training_data_handler()
        pd.set_option('display.expand_frame_repr', False)

        # encode classes
        self.one_hot = MultiLabelBinarizer()
        # import IPython
        # IPython.embed()
        self.Y = self.one_hot.fit_transform(self.df['Y'])

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
        print("Ready to train")

    # Builds the tokenizer depending on your training dataset
    def build_tokenizer(self):
        self.tokenizer.fit_on_texts(self.df['text'])
        self.vocabulary_size = len(self.tokenizer.word_index) + 1
        with open(MODEL_PATH+'tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Saves the one_hot encoder. This depends on your class label
    def save_encoder(self):
        # save encoder obj, can be used for prediction
        with open(MODEL_PATH+'onehot.pickle', 'wb') as handle:
            pickle.dump(self.one_hot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return word embedding matrix using glove for the first layer of the NN
    def create_embedding_matrix(self):
        embeddings_index = dict()
        f = open(GLOVE_PATH, encoding="utf8")
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

    # function to start training the model
    def train_model(self,model_file=MODEL_NAME):
        if not self.ignore:
            self.history = self.model.fit(self.padded_sent, self.Y, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose = VERBOSE)
            self.model.save(MODEL_PATH + model_file)
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

    # Only the functions below this point vary between classifiers in Train class

    # Function for handling class labels based on type of classifier used
    # and dropping all class labels except training class label which is renamed to 'Y'
    def class_label_handler(self):
        self.df['Y'] = self.df[TRAINING_LABEL].apply(self.class_label_processing)
        self.df = self.df.drop(['label_SALLY', 'label_struck', 'label_Frenard'], axis=1)
        self.df = self.df.dropna()

    # This handles the different ways each classifier requires the desired class labels
    @staticmethod
    def class_label_processing(class_label):
        if MODEL_TYPE == 'SentenceClassifier':
            if 'U' in class_label:
                return None
            else:
                return class_label.split(',')[0]
        elif MODEL_TYPE == 'MultiClassSentenceClassifier':
            if 'U' in class_label:
                return None
            else:
                return class_label
        elif MODEL_TYPE == 'BinarySentenceClassifier':
            if 'U' in class_label:
                return 'U'
            else:
                return 'N'

    # This is used to add/drop or modify the data-frame depending on the classifier/ needs of user
    def training_data_handler(self):
        if MODEL_TYPE == 'SentenceClassifier':
            pass
        elif MODEL_TYPE == 'MultiClassSentenceClassifier':
            pass
        elif MODEL_TYPE == 'BinarySentenceClassifier':
            useful_sentences = self.df[self.df['Y'] == 'N']
            self.df = self.df[self.df['Y'] == 'U'][:len(useful_sentences)].append(useful_sentences)
            msk = np.random.rand(len(self.df)) < 0.5
            tmp = self.df[msk]
            # import IPython
            # IPython.embed()
            self.df = tmp.append(self.df[~msk])

    # Generates model structure depending on type of classification performed
    def generate_model(self):
        if MODEL_TYPE == 'SentenceClassifier':
            return self.generate_sentence_classifier_model()
        elif MODEL_TYPE == 'MultiClassSentenceClassifier':
            return self.generate_multi_class_sentence_classifier_model()
        elif MODEL_TYPE == 'BinarySentenceClassifier':
            return self.generate_binary_sentence_classifier_model()

    # Generates model for SentenceClassifier
    def generate_sentence_classifier_model(self):
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, 100, input_length=self.sentence_length, weights=[self.embedding_matrix],
                            trainable=True))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(self.lstm_size))
        model.add(Dense(self.number_of_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Generates model for MultiClassSentenceClassifier
    def generate_multi_class_sentence_classifier_model(self):
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, 100, input_length=self.sentence_length, weights=[self.embedding_matrix],
                            trainable=True))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(self.lstm_size))
        model.add(Dense(self.number_of_classes, activation='sigmoid'))
        # at least one class match
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Generates model for BinarySentenceClassifier
    def generate_binary_sentence_classifier_model(self):
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, 100, input_length=self.sentence_length, weights=[self.embedding_matrix], trainable=True))
        model.add(Conv1D(256,5, activation='relu'))
        # model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(256,5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Flatten())
        # model.add(LSTM(self.lstm_size))
        model.add(Dense(self.number_of_classes,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class Predict(object):
    # hyper params used shoould be consistent to those used while training.
    def __init__(self):
        self.sentence_length = SENTENCE_LENGTH
        self.confidence = PRED_CONFIDENCE  # used only in MultiClassSentenceClassifier
        self.model = None
        self.tokenizer = None
        self.one_hot = None

    def load_model(self):
        # input a sentence or string
        # return a set of tags in the input
        # load model, tokenizer, and encoder
        self.model = load_model(MODEL_PATH + MODEL_NAME)
        with open(MODEL_PATH+'tokenizer.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(MODEL_PATH+'onehot.pickle', 'rb') as o:
            self.one_hot = pickle.load(o)

    # Function to predict the data and modify data-frame
    def predict_data(self,test_data=TEST_DATA):
        # Load test data
        self.df = pd.read_hdf(PROCESSED_DATA_PATH + test_data, key='test')
        self.class_label_handler()
        tmp = self.df['text']
        self.predicted_vals = self.df['text'].apply(self.predict)
        self.df['text'] = tmp
        self.df['Predicted_Values'] = self.predicted_vals

    # Only functions from below this point vary between classifiers

    # Predicts label of individual sentence. label returned depends on type of classifier used
    def predict(self, text):
        text = [text]
        # encode and pad
        encoded_sent = self.tokenizer.texts_to_sequences(text)
        padded_sent = pad_sequences(encoded_sent, self.sentence_length, padding='post')

        # predict and decode to original tags(classes)
        pred = self.model.predict(padded_sent)

        if MODEL_TYPE == 'SentenceClassifier' or MODEL_TYPE == 'BinarySentenceClassifier':
            tags = self.one_hot.inverse_transform(pred == pred.max())[0][0]
        elif MODEL_TYPE == 'MultiClassSentenceClassifier':
            y_classes = (pred > PRED_CONFIDENCE).astype(int)
            tags = self.one_hot.inverse_transform(y_classes)[0]
        return tags

    # Uses same class handler as training data. If you modify this function, make sure you do the same in the Train class
    def class_label_handler(self):
        self.df['Y'] = self.df[TRAINING_LABEL].apply(Train.class_label_processing)
        self.df = self.df.drop(['label_SALLY', 'label_struck', 'label_Frenard'], axis=1)
        self.df = self.df.dropna()

    # This returns the accuracy of the model with test data. Also saves the analysis in a resulting csv folder
    def analyze(self):
        if MODEL_TYPE == 'MultiClassSentenceClassifier':
            total_matching = 0
            for i in self.df.index:
                tmp = self.df['Y'][i].split(',')
                for j,v in enumerate(tmp):
                    tmp[j] = v.strip()
                tmp = set(tmp)
                self.predicted_vals[i] = set(self.predicted_vals[i])
                self.df['Y'][i] = tmp
        else:
            pass
        total_matching = sum(self.df['Y'] == self.predicted_vals)
        total = len(self.predicted_vals)
        print('accuracy = ',total_matching/total)
        self.df.to_csv(MODEL_PATH + PREDICTION_FILE, header=True)
