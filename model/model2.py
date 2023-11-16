#model2: 
#predicting the approximate number of the scanned receipts for each month of 2022
#based on the difference between consecutive months

#pandas
#numpy
#tensorflow
#pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


# Provided data
def train_model2(data_path):
    df = pd.read_csv(data_path)
    print(df.head(5))
    print(df.columns)
    print(df.dtypes)
    df['# Date'] = pd.to_datetime(df['# Date'])
    df['Month'] = df['# Date'].dt.month
    df['Year'] = df['# Date'].dt.year

    # Feature engineering
    df_agg = df.groupby(['Year', 'Month'], as_index=False)['Receipt_Count'].sum()


    # Take the difference between consecutive values
    df_agg['Diff_Values'] = df_agg['Receipt_Count'].diff().fillna(0)

    # Create input and target data for training
    X_train = df_agg['Diff_Values'][:-1].values.reshape(-1, 1, 1)
    y_train = df_agg['Diff_Values'][1:].values.reshape(-1, 1)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=1, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=2)

    # Generate predictions for the next 12 values
    X_pred = df_agg['Diff_Values'].values[-1].reshape(1, 1, 1)
    y_pred_diff = []
    for _ in range(12):
        pred_diff = model.predict(X_pred)
        y_pred_diff.append(pred_diff.flatten())
        X_pred = pred_diff.reshape(1, 1, 1)

    print("1:",df_agg['Diff_Values'].values)
    print("2:", y_pred_diff)
    res = np.concatenate(y_pred_diff)
    print("3:", res)
    # Cumulatively sum the differences to obtain predictions for the original time series
    y_pred_cumsum = np.cumsum(np.concatenate([df_agg['Diff_Values'].values, res]))
                                            
    # print(y_pred_cumsum)
                                            
    predictions = []
    dec2021_val = 309948684
    i = 1
    j = 1
    fir = 309948684 + res[0]
    predictions.append(fir)
    while(i<12 and j<12):
        predictions.append(predictions[i-1] + y_pred_cumsum[j])
        i+=1
        j+=1
    print(predictions)

    # Save the predictions to a file using pickle
    with open('output/model2_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
