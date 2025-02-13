import pandas as pd
import numpy as np
import os
path = '../model-based-mbr/PreferenceSets'
pathTo = '../model-based-mbr/PreferenceSetsSplits'
for p in os.listdir(path):
    file =  os.path.join(path,p)
    df = pd.read_csv(file)

    shuffled_df = df.iloc[np.random.permutation(len(df))]

    split_index = int(len(shuffled_df) * 0.9)
    df_first_90 = shuffled_df.iloc[:split_index]
    df_last_10 = shuffled_df.iloc[split_index:]

    train_file = p.split('.csv')[0]+'_train.csv'
    train_file = os.path.join(pathTo,train_file)

    valid_file = p.split('.csv')[0]+'_test.csv'
    valid_file = os.path.join(pathTo,valid_file)

    df_last_10.to_csv(valid_file, index=False)
    df_first_90.to_csv(train_file, index=False)