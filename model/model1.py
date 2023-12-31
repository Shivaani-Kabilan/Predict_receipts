#model1: 
#predicting the approximate number of the scanned receipts for each month of 2022
#based on the previous month

#pandas
#numpy
#tensorflow
#pickle

#!pip3 install pandas
#!pip3 install tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Assuming you have a dataset like the one you provided
# "Date" column should be in datetime format
# "Receipt_Count" column represents the number of receipts received on each day

# Load dataset
def train_model1(data_path):
    df = pd.read_csv(data_path)
    print(df.head(5))
    print(df.columns)
    print(df.dtypes)
    df['# Date'] = pd.to_datetime(df['# Date'])
    df['Month'] = df['# Date'].dt.month
    df['Year'] = df['# Date'].dt.year

    # Feature engineering
    df_agg = df.groupby(['Year', 'Month'], as_index=False)['Receipt_Count'].sum()
    df_agg['Prev_Month_Receipts'] = df_agg['Receipt_Count'].shift(1)
    
    monthy_2021 = df_agg['Receipt_Count'].values

    # Drop the first row as it will have NaN due to shift
    df_agg = df_agg.dropna()

    # Model data preparation
    X = df_agg[['Prev_Month_Receipts']].values
    y = df_agg['Receipt_Count'].values

    # Normalize data manually
    X_mean, X_std = X.mean(), X.std()
    X_normalized = (X - X_mean) / X_std

    y_mean, y_std = y.mean(), y.std()
    y_normalized = (y - y_mean) / y_std

    # Split data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(X_normalized) * split_ratio)

    X_train, X_test = X_normalized[:split_index], X_normalized[split_index:]
    y_train, y_test = y_normalized[:split_index], y_normalized[split_index:]
    
    split_ind = int(len(X_train) * split_ratio)
    
    X_train, X_val = X_train[:split_ind], X_train[split_ind:]
    y_train, y_val = y_train[:split_ind], y_train[split_ind:]
    

    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=1, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer, loss='mean_squared_error', metrics=['mae'])

    # Training
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=8, verbose=2)
    epochs = 100
    # Extracting MSE and MAE metrics from history
    training_mse = history.history['loss']
    validation_mse = history.history['val_loss']
    training_mae = history.history['mae']
    validation_mae = history.history['val_mae']

    # Plotting the metrics
    epochs_range = range(1, epochs + 1)

    # Plotting the data
    matplotlib.pyplot.switch_backend('Agg') 
    plt.figure(figsize=(10, 4))

    # Plot MSE
    #plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_mse, label='Training MSE')
    plt.plot(epochs_range, validation_mse, label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.savefig('static/metrics1.png', format='png')
    plt.close()

    # Plot MAE
    plt.figure(figsize=(10, 4))
    #plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_mae, label='Training MAE')
    plt.plot(epochs_range, validation_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()

    plt.savefig('static/metrics2.png', format='png')

    plt.close()

    # Evaluation
    loss = model.evaluate(X_test, y_test)
    print(f'Mean Squared Error on Test Set: {loss}')

    # Inference for the year 2022
    # Assuming new_data is the input for each day's receipts in 2021
    # predicted_normalized = model.predict(X_normalized).flatten()
    # predicted = predicted_normalized * y_std + y_mean

    # # Print the predicted number of receipts for each month of 2022
    # print(predicted)


    # Generate predictions for the next 12 values
    future_values = []

    # Use the last sequence of known values to predict the next value

    last_sequence = X_normalized[-1]
    for _ in range(12):
        prediction = model.predict(last_sequence)
        future_values.append(prediction.flatten())
        last_sequence = prediction.flatten()
    

    # Denormalize the predictions
    future_values = np.array(future_values) * y_std + y_mean

    print(future_values)

    # Save the predictions to a file using pickle
    with open('output/model1_predictions.pkl', 'wb') as file:
        pickle.dump(future_values, file)

    # Save the 2021 receipts to a file using pickle
    with open('output/2021receipts.pkl', 'wb') as g:
        pickle.dump(monthy_2021, g)