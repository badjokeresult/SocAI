import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train(self, criterion, optimizer, data, epochs=50):
        for _ in range(epochs):
            super().train()
            optimizer.zero_grad()
            outputs = self(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
    
    def test(self, data):
        test_data = torch.tensor(data, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            reconstructed = self(test_data)
            mse = torch.mean((test_data - reconstructed)**2, dim=1).numpy()
        normal_mse = mse[:1000]
        threshold = np.mean(normal_mse) + 3 * np.std(normal_mse)
        return threshold, mse

    def detect(self, mse, threshold):
        return (mse > threshold).astype(int)