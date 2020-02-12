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
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Embedding, Dropout, MaxPooling1D, LSTM
# from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import load_model


# This class handles all the training portion of classifiers using neural networks
class Train(object):
    # The init function performs all the tasks in a sequential manner to perform training
    def __init__(self, force_retrain=False, training_data=TRAIN_DATA):
        # Check if model path exists and create it if it doesn't
        if not os.path.exists(PROCESSED_DATA_PATH):
            os.mkdir(PROCESSED_DATA_PATH)
        else:
            # Check if model file exists and no RETRAIN is set to false to skip training
            if os.path.exists(PROCESSED_DATA_PATH + MODEL_NAME) and not force_retrain:
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
        with open(PROCESSED_DATA_PATH+'tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Saves the one_hot encoder. This depends on your class label
    def save_encoder(self):
        # save encoder obj, can be used for prediction
        with open(PROCESSED_DATA_PATH + 'onehot.pickle', 'wb') as handle:
            pickle.dump(self.one_hot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # embedding matrix
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

    def train_model_with_kfold(self):
        X = self.padded_sent
        Y = self.Y
        n_split = 10
        k_fold_train_accuracy = []
        k_fold_val_accuracy = []
        k_fold_test_accuracy = []

        for train_index, test_index in StratifiedShuffleSplit(n_split, random_state=0).split(X, Y):
        #for train_index, test_index in ShuffleSplit(n_split, random_state=0).split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            train_acc, val_acc, test_acc = self.train_model_for_every_k(x_train, y_train,x_test, y_test)

            k_fold_train_accuracy.append(train_acc)
            k_fold_val_accuracy.append(val_acc)
            k_fold_test_accuracy.append(test_acc)

        x = [1,2,3,4,5,6,7,8,9,10]
        print("K_fold Training Accuracy for k = 10 fold : ", k_fold_train_accuracy)
        print("K_fold Validation Accuracy for k = 10 fold : ", k_fold_val_accuracy)
        print("K_fold test Accuracy for k = 10 fold : ", k_fold_test_accuracy)

        plt.plot(x, k_fold_train_accuracy)
        plt.plot(x, k_fold_val_accuracy)
        plt.plot(x, k_fold_test_accuracy)
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        plt.yticks(np.arange(0.0, 1.0 + 0.2, 0.10))
        plt.title('K Fold accuracy GloVe Multi Class - Sally-Frenard-agree')
        plt.ylabel('accuracy')
        plt.xlabel('Fold Value')
        plt.legend(['train', 'validation', 'test'], loc='lower right')
        plt.savefig("K Fold accuracy GloVe Multi Class - Sally-Frenard-agree.png", dpi=1200)
        plt.show()



    def train_model_for_every_k(self, x_train, y_train, x_test, y_test):
        model_file = MODEL_NAME
        model = Sequential()
        model.add(
            Embedding(self.vocabulary_size, 100, input_length=self.sentence_length, weights=[self.embedding_matrix],
                      trainable=True))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        # model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(self.lstm_size))
        model.add(Dense(self.number_of_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if not self.ignore:
            self.history = model.fit(x_train, y_train, validation_split=0.1, batch_size=512, epochs=ep, verbose =1)
            model.save(PROCESSED_DATA_PATH + model_file)
            #self.plot_history()
            test_loss, test_acc = model.evaluate(x_test, y_test)
        return max(self.history.history['acc']), max(self.history.history['val_acc']), test_acc

    def train_model(self, model_file=MODEL_NAME):
        if not self.ignore:
            self.history = self.model.fit(self.padded_sent, self.Y, validation_split=0.1, batch_size=512, epochs=ep,
                                          verbose=1)
            self.model.save(PROCESSED_DATA_PATH + model_file)
            self.plot_history()
        return max(self.history.history['acc']), max(self.history.history['val_acc'])

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
        # #For inter coder reliability
        # self.df['Y'] = self.df[TRAINING_LABEL].apply(self.class_label_processing)
        # self.df = self.df.drop(['label_SALLY', 'label_struck', 'label_Frenard'], axis=1)
        # self.df = self.df.dropna()

        # For Sally Dataset
        self.df['Y'] = self.df['Y'].apply(self.class_label_processing)

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
            self.df = self.df.dropna()
            pass
        elif MODEL_TYPE == 'MultiClassSentenceClassifier':
            self.df = self.df.dropna()
            pass
        elif MODEL_TYPE == 'BinarySentenceClassifier':
            useful_sentences = self.df[self.df['Y'] == 'N']
            self.df = self.df[self.df['Y'] == 'U'][:len(useful_sentences)].append(useful_sentences)
            msk = np.random.rand(len(self.df)) < 0.5
            tmp = self.df[msk]
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
        # model.add(MaxPooling1D(pool_size=4))
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
        model.add(Embedding(self.vocabulary_size, 100, input_length=self.sentence_length, weights=[self.embedding_matrix],trainable=True))
        model.add(Dropout(0.2))
        # model.add(Conv1D(64,5, activation='relu'))
        # model.add(MaxPooling1D(pool_size=4))
        # model.add(Conv1D(256,5, activation='relu'))
        # model.add(MaxPooling1D(pool_size=4))
        # model.add(Flatten())
        model.add(LSTM(self.lstm_size))
        model.add(Dense(self.number_of_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class Predict(object):
    # hyper params, should use the same as used in train
    def __init__(self,data_file='predict.h5'):
        self.sentence_length = SENTENCE_LENGTH
        self.confidence = PRED_CONFIDENCE  # suggested 0.5, used only in MultiClassSentenceClassifier
        self.model = None
        self.tokenizer = None
        self.one_hot = None
        self.train = Train()

    def load_model(self):
        # input a sentence or string
        # return a set of tags in the input
        # load model, tokenizer, and encoder
        self.model = load_model(PROCESSED_DATA_PATH + MODEL_NAME)
        with open(PROCESSED_DATA_PATH+'tokenizer.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(PROCESSED_DATA_PATH+'onehot.pickle', 'rb') as o:
            self.one_hot = pickle.load(o)

    # Function to predict the data and modify data-frame
    def predict_data(self,test_data=TEST_DATA):
        self.df = pd.read_hdf(PROCESSED_DATA_PATH + test_data, key='test')
        self.class_label_handler()
        tmp =self. df['text']
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
        # #For inter coder reliability
        # self.df['Y'] = self.df[TRAINING_LABEL].apply(Train.class_label_processing)
        # self.df = self.df.drop(['label_SALLY', 'label_struck', 'label_Frenard'], axis=1)
        # self.df = self.df.dropna()

        # For Sally Dataset
        self.df['Y'] = self.df['Y'].apply(Train.class_label_processing)
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
        self.df.to_csv( PROCESSED_DATA_PATH+ PREDICTION_FILE, header=True)
        accuracy = total_matching / total
        return self.df, accuracy
