from torch import nn


class DeepLog(nn.Module):
    """
    DeepLog model using LSTM for anomaly detection in log sequences.
    """

    def __init__(self, embedding_dim, hidden_size, num_layers, vocab_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer for input log keys
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM network for sequence modeling
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)

        # Fully connected layer for output predictions
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        """
        Forward pass through DeepLog model.
        """
        embedded = self.embeddings(x)  # Convert input to embeddings
        out, (hidden, _) = self.lstm(embedded)  # Pass through LSTM
        output = self.fc(hidden)  # Fully connected layer output
        return output.squeeze(0)  # Remove extra dimension
