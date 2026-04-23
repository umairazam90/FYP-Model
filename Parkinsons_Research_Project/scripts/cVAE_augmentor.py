
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.notebook import tqdm
import numpy as np

class cVAE(nn.Module):
    '''A Conditional VAE for tabular data
    It learns P(X | y), so we can generate data by *specifying* the label.
    '''
    def __init__(self, input_dim, label_dim=1, latent_dim=6):
        super(cVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.label_dim = label_dim

        # Encoder
        # Input is now (features + label)
        self.enc_fc1 = nn.Linear(input_dim + label_dim, 24)
        self.enc_fc2 = nn.Linear(24, 12)
        self.enc_fc_mean = nn.Linear(12, latent_dim)
        self.enc_fc_logvar = nn.Linear(12, latent_dim)

        # Decoder
        # Input is now (latent vector + label)
        self.dec_fc1 = nn.Linear(latent_dim + label_dim, 12)
        self.dec_fc2 = nn.Linear(12, 24)
        self.dec_fc3 = nn.Linear(24, input_dim)

    def encode(self, x, y):
        # Concatenate features and label
        inputs = torch.cat([x, y], dim=1)
        h = F.relu(self.enc_fc1(inputs))
        h = F.relu(self.enc_fc2(h))
        return self.enc_fc_mean(h), self.enc_fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # Concatenate latent vector and label
        inputs = torch.cat([z, y], dim=1)
        h = F.relu(self.dec_fc1(inputs))
        h = F.relu(self.dec_fc2(h))
        return self.dec_fc3(h) # No sigmoid, data is standardized

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, self.input_dim), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_cvae(cvae, data_loader, device, epochs=100):
    cvae.to(device)
    cvae.train()
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)

    print(f"--- Training cVAE for {epochs} epochs ---")
    for epoch in tqdm(range(epochs), desc=f"Training cVAE", leave=False):
        for (data, labels) in data_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = cvae(data, labels)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
    print(f"✅ cVAE Training Complete.")
    return cvae

def generate_synthetic_samples_conditional(cvae, num_samples, label_value, input_dim, latent_dim, device):
    cvae.eval()
    with torch.no_grad():
        # Sample from the prior distribution (standard normal)
        z = torch.randn(num_samples, latent_dim).to(device)
        # Create the desired label
        y_label = torch.full((num_samples, 1), float(label_value)).to(device)
        synthetic_data = cvae.decode(z, y_label)
        return synthetic_data.cpu().numpy()
