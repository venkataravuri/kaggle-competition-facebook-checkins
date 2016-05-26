import pandas as pd
import numpy as np
from sklearn import cross_validation
import util
from config import Config


@util.timeit
def load_train_data():
    df_train = pd.read_csv(Config.train_data_file, header=0, index_col=0)
    print("Training dataset shape:", df_train.shape)
    print("Training dataset first 5 records:")
    print(df_train.head(5))

    return df_train


@util.timeit
def train_test_split(df_train):
    # Holding out 40% of the data for testing or evaluating classifier(s)
    X_df_train, X_df_test, y_df_train, y_df_test = cross_validation.train_test_split(
        df_train[['x', 'y', 'accuracy', 'time']], pd.DataFrame(df_train['place_id']),
        test_size=0.4)

    print("Cross validation training dataset shape:", X_df_train.shape)
    print(X_df_train.head(2))
    print("Cross validation test dataset shape:", X_df_test.shape)
    print(X_df_test.head(2))
    print("Cross validation training dataset actual result shape:", y_df_train.shape)
    print(y_df_train.head(2))
    print("Cross validation test dataset actual result shape:", y_df_test.shape)
    print(y_df_test.head(2))

    return X_df_train, X_df_test, y_df_train, y_df_test
