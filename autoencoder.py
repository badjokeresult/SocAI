import torch
import torch.nn as nn


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
    
    def fit(self, criterion, optimizer, data, epochs=50, batch_size=128, verbose=True):
        self.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.int64))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in loader:
                inputs = batch[0]
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def detect(self, mse, threshold):
        return (mse > threshold).astype(int)