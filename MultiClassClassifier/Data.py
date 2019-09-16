import os
import glob
import numpy as np
import pandas as pd
import string
from Config import *


class CreateDataset(object):
    def __init__(self):
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
            if len(file.split('-')) == 1:
                file_list.append(file)
        return file_list

    def process_raw_to_hd(self):
        filenames = self.get_file_names()
        main_df = pd.DataFrame()
        for i in filenames:
            df = pd.read_csv(i)
            # print(df)
            df['text'] = df['text'].apply(str)
            main_df = pd.concat([main_df , df], ignore_index=True)
        if not os.path.exists(PROCESSED_DATA_PATH):
            os.mkdir(PROCESSED_DATA_PATH)
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
        df = pd.read_hdf(PROCESSED_DATA_PATH + RAW_DATA,key='raw')
        df = df.dropna()
        df['text'] = df['text'].apply(self.clean_text)
        df['Y'] = df[TRAINING_LABEL].apply(self.class_label_handler)
        df = df.drop(['label_SALLY', 'label_struck', 'label_Frenard'], axis=1)
        df = df.dropna()
        # import IPython
        # IPython.embed()
        df.to_hdf(PROCESSED_DATA_PATH + CLEAN_DATA, key='clean')#, format='t', min_itemsize={'text': 4096})

    @staticmethod
    def class_label_handler(class_label):
        if 'U' in class_label:
            return None
        else:
            return class_label.split(',')[0]


    @staticmethod
    def split_train_test():
        df = pd.read_hdf(PROCESSED_DATA_PATH + CLEAN_DATA,key='clean')
        msk = np.random.rand(len(df)) < TRAIN_TEST_SPLIT
        train = df[msk]
        test = df[~msk]
        train.to_hdf(PROCESSED_DATA_PATH + TRAIN_DATA, key='train')#, format='t', min_itemsize={'text': 4096})
        test.to_hdf(PROCESSED_DATA_PATH + TEST_DATA, key='test')#   , format='t', min_itemsize={'text': 4096})


