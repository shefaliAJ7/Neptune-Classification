import pandas as pd

df = pd.read_hdf('processed_data/clean_data.h5',key='clean')
count = sum(df['class'])
print(count/len(df))

