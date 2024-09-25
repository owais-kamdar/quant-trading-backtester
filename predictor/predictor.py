# predictor.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_data_lstm(data, n_steps=50):
    features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Middle']
    data = data[features].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
        y.append(scaled_data[i, 0])  # Predicting 'Close' price

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def predict_future_prices(data, n_steps=50, epochs=20, batch_size=32, device='cpu'):
    """
    Predict future stock prices using PyTorch LSTM.
    Returns predicted future prices and evaluation metrics.
    """
    # Prepare data
    X, y, scaler = prepare_data_lstm(data, n_steps)
    input_size = X.shape[2]  # Number of features

    # Split data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Build the model
    model = LSTMModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.view(-1), y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}')

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions.view(-1), y_test)
        test_predictions_np = test_predictions.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        # Calculate additional metrics
        mae = mean_absolute_error(y_test_np, test_predictions_np)
        rmse = np.sqrt(mean_squared_error(y_test_np, test_predictions_np))
        print(f'Test Loss (MSE): {test_loss.item():.4f}')
        print(f'Test MAE: {mae:.4f}')
        print(f'Test RMSE: {rmse:.4f}')

    # Predict future prices (for the next 10 days)
    # Ensure last_data is a PyTorch tensor
    last_data = X[-1]
    last_data = torch.tensor(last_data, dtype=torch.float32).to(device).unsqueeze(0)
    predicted_future = []

    with torch.no_grad():
        for _ in range(10):
            future_price = model(last_data)
            predicted_future.append(future_price.cpu().item())
            # Prepare next input
            # Get the last feature vector
            last_feature_vector = last_data[:, -1, :]  # Shape: [1, input_size]

            # Replace the 'Close' price (assumed to be at index 0) with the predicted future price
            future_feature_vector = last_feature_vector.clone()
            future_feature_vector[:, 0] = future_price  # Replace 'Close' price

            # Add a time dimension to make it [1, 1, input_size]
            future_feature_vector = future_feature_vector.unsqueeze(1)  # Shape: [1, 1, input_size]

            # Update last_data
            last_data = torch.cat((last_data[:, 1:, :], future_feature_vector), dim=1)

    # Rescale back to original price scale
    # Prepare data for inverse transformation
    predicted_future_scaled = np.zeros((len(predicted_future), data.shape[1]))  # Number of features
    predicted_future_scaled[:, 0] = predicted_future  # Assuming 'Close' is the first feature

    predicted_future_prices = scaler.inverse_transform(predicted_future_scaled)[:, 0]

    return predicted_future_prices, {'Test Loss': test_loss.item(), 'MAE': mae, 'RMSE': rmse}
