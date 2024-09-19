import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def prepare_data_lstm(data, n_steps=50):
    """
    Prepares stock data for LSTM.
    :param data: DataFrame with stock prices.
    :param n_steps: Number of timesteps to look back for LSTM.
    :return: Scaled input features (X), target prices (y), and the scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape, units=50):
    """
    Builds and compiles the LSTM model.
    :param input_shape: Shape of the input data for LSTM.
    :param units: Number of units in the LSTM layers.
    :return: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=units))
    model.add(Dense(1))  # Output layer for future price
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(data, n_steps=50, epochs=10, batch_size=32):
    """
    Predict future stock prices using LSTM.
    :param data: DataFrame with stock prices.
    :param n_steps: Number of timesteps to look back for LSTM.
    :param epochs: Number of epochs to train the model.
    :param batch_size: Batch size for model training.
    :return: Predicted future stock prices.
    """
    # Prepare data for LSTM
    X, y, scaler = prepare_data_lstm(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    model = build_lstm_model((X.shape[1], 1))

    # Save model during training
    checkpoint_filepath = 'best_model.h5'
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True)

    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1,
              callbacks=[early_stopping, model_checkpoint])

    model.load_weights(checkpoint_filepath)

    # Predict future prices (for the next 10 days as an example)
    last_data = X[-1].reshape((1, X.shape[1], 1))  # Use the last data point for prediction
    predicted_future = []

    for _ in range(10):  # Predict for 10 days ahead
        future_price = model.predict(last_data)
        predicted_future.append(future_price[0][0])
        # Append predicted value and shift for future prediction
        last_data = np.append(last_data[:, 1:, :], future_price.reshape(1, 1, 1), axis=1)

    # Rescale back to original price scale
    predicted_future = scaler.inverse_transform(np.array(predicted_future).reshape(-1, 1))
    
    return predicted_future
