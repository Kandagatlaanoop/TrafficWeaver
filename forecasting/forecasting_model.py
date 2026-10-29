# forecasting/forecasting_model.py
import torch
import torch.nn as nn

class TrafficForecaster(nn.Module):
    """
    Defines the LSTM model architecture using PyTorch.
    This model takes a sequence of past traffic data and predicts future values.
    """
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=4):
        """
        Args:
            input_size (int): Number of input features (e.g., traffic volume for 4 approaches).
            hidden_layer_size (int): Number of neurons in the LSTM hidden layer.
            output_size (int): Number of predicted features.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # Define the fully connected output layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        """Defines the forward pass of the model."""
        lstm_out, _ = self.lstm(input_seq)
        # We take the output of the last time step to make the prediction
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions