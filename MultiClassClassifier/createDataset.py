import os
import glob
import numpy as np
import pandas as pd
import IPython
import string
import Config


class createDataset:
    def __init__(self):
        if not os.path.exists(Config.PROCESSED_DATA_PATH + 'raw_data.h5'):
            print("Processing Raw Data...")
            self.process_raw_to_hd()

    def get_file_names(self):
        files = [f for f in glob.glob(Config.CSV_DATA_PATH+"*.csv")]
        compr_files = []
        for file in files:
            if len(file.split('-')) == 1:
                compr_files.append(file)
        return compr_files

    def process_raw_to_hd(self):
        filenames = self.get_file_names()
        main_df = pd.DataFrame()
        for i in filenames:
            df = pd.read_csv(i)
            print(df)
            main_df = pd.concat([main_df , df], ignore_index=True)
        os.mkdir(Config.PROCESSED_DATA_PATH)
        main_df.to_hdf(Config.PROCESSED_DATA_PATH+'raw_data.h5', key='raw', append=True, format='t', min_itemsize={'text': 4096})
        return main_df

    def clean_text(self,line):
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
