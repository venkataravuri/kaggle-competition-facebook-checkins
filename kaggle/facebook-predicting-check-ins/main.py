import loaddata

if __name__ == '__main__':
    df_train = loaddata.load_train_data()
    X_df_train, X_df_test, y_df_train, y_df_test = loaddata.train_test_split(df_train)
