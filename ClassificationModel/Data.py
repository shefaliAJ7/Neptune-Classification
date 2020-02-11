import os
import glob
import numpy as np
import pandas as pd
import string
from Config import *


class CreateDataset(object):
    total_data = None
    count_usefull = 0
    def __init__(self):
        self.total_data = 0
        self.count_usefull = 0
        if not os.path.exists(PROCESSED_DATA_PATH + RAW_DATA):
            self.process_raw_to_hd()
        if not os.path.exists(PROCESSED_DATA_PATH + CLEAN_DATA):
            self.clean_data()
        if not os.path.exists(PROCESSED_DATA_PATH + TRAIN_DATA) or not os.path.exists(PROCESSED_DATA_PATH + TEST_DATA):
            self.split_train_test()


    @staticmethod
    def get_file_names():
        files = [f for f in glob.glob(CSV_DATA_PATH+"*.csv")]
        file_list = []
        for file in files:
            file_list.append(file)
        return file_list

    def process_raw_to_hd(self):
        filenames = self.get_file_names()
        main_df = pd.DataFrame()
        for i in filenames:
            df = pd.read_csv(i)
            df['text'] = df['text'].apply(str)
            main_df = pd.concat([main_df , df], ignore_index=True)
        if not os.path.exists(PROCESSED_DATA_PATH):
            os.mkdir(PROCESSED_DATA_PATH)
        print(main_df.info())
        main_df.to_hdf(PROCESSED_DATA_PATH + RAW_DATA, key='raw', append=True, format='t', min_itemsize={'text': 4096})
        return main_df

    @staticmethod
    def clean_text(line):
        # Converting to lower
        line = line.lower()

        # Removing alphanumerics
        tokens = [word for word in line.split() if word.isalpha()]

        # Removing Punctuations
        translator = str.maketrans("", "", string.punctuation)
        tokens = [word.translate(translator) for word in tokens]

        # Removing stop_words
        # stop_words = set(stopwords.words('english'))
        # tokens = [word for word in tokens if not word in stop_words]

        # Removing short_words
        tokens = [word for word in tokens if len(word) > 1]
        return tokens

    def clean_data(self):
        df = pd.read_hdf(PROCESSED_DATA_PATH + RAW_DATA)
        df = df.dropna()
        df['text'] = df['text'].apply(self.clean_text)
        if MODEL_TYPE == 'BinarySentenceClassifier':
            self.count_usefull = 0
            for label in df[TRAINING_LABEL[0]]:
                if label != 'U':
                    self.count_usefull += 1
            print("No. of Usefull Sentences: " + str(self.count_usefull))
            print()
            df['Y'] = df[TRAINING_LABEL[0]].apply(self.class_label_handler_binarysentenceclassifier)
        else:
            if len(TRAINING_LABEL) == 1:
                df['Y'] = df[TRAINING_LABEL[0]].apply(self.class_label_handler)
            else:
                for label in TRAINING_LABEL:
                    df[label] = df[label].apply(self.class_label_handler)
                if AGREEMENT == 1:
                    df['Y'] = df.apply(self.get_similar, axis=1)
                elif AGREEMENT == 0:
                    df['Y'] = df.apply(self.get_different, axis=1)

        df = df.drop(TRAINING_LABEL_DROP, axis=1)
        df = df.dropna()
        self.total_data = len(df['text'])
        # import IPython
        # IPython.embed()
        df.to_hdf(PROCESSED_DATA_PATH + CLEAN_DATA, key='clean')#, format='t', min_itemsize={'text': 4096})

    def class_label_handler_binarysentenceclassifier(self, class_label):
        if class_label == 'U':
            self.count_usefull -= 1
            if self.count_usefull < 0:
                return np.NaN
        return class_label

    @staticmethod
    def class_label_handler(class_label):
        if 'U' in class_label:
            return np.NaN
        else:
            return class_label.split(',')[0]

    @staticmethod
    def get_similar(row):
        if len(TRAINING_LABEL)==2:
            if row[TRAINING_LABEL[0]] == row[TRAINING_LABEL[1]] and row[TRAINING_LABEL[0]] != 'U':
                return row[TRAINING_LABEL[0]]
            else:
                return None
        if len(TRAINING_LABEL)==3:
            if row[TRAINING_LABEL[0]] == row[TRAINING_LABEL[1]] == row[TRAINING_LABEL[2]] and row[TRAINING_LABEL[0]] != 'U':
                return row[TRAINING_LABEL[0]]
            else:
                return np.NaN

    @staticmethod
    def get_different(row):
        if len(TRAINING_LABEL) == 2:
            if row[TRAINING_LABEL[0]] == 'U' or row[TRAINING_LABEL[0]] == row[TRAINING_LABEL[1]]:
                return np.NaN
            else:
                return row[TRAINING_LABEL[0]]
        if len(TRAINING_LABEL) == 3:
            if row[TRAINING_LABEL[0]] == 'U' or row[TRAINING_LABEL[0]] == row[TRAINING_LABEL[1]] == row[TRAINING_LABEL[2]]:
                return np.NaN
            else:
                return row[TRAINING_LABEL[0]]


    @staticmethod
    def split_train_test():
        df = pd.read_hdf(PROCESSED_DATA_PATH + CLEAN_DATA,key='clean')
        msk = np.random.rand(len(df)) < TRAIN_TEST_SPLIT
        train = df[msk]
        test = df[~msk]
        train.to_hdf(PROCESSED_DATA_PATH + TRAIN_DATA, key='train')#, format='t', min_itemsize={'text': 4096})
        test.to_hdf(PROCESSED_DATA_PATH + TEST_DATA, key='test')#   , format='t', min_itemsize={'text': 4096})


