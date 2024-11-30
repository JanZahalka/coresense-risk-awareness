"""
ml.autoencoder

The autoencoder models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from data import ml_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VariationalAutoEncoder(nn.Module):
    """
    A variational autoencoder.
    """

    IMG_SIZE = 128

    TRANSFORM = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)  # Mean of latent space
        self.logvar = nn.Linear(256, latent_dim)  # Log variance of latent space

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def encode(self, x):
        """
        Encode the input data.
        """
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode the output from the latent space.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass of the autoencoder.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def loss(recon_x, x, mu, logvar):
        """
        The classic loss function for the VAE: reconstruction loss + Kullback-Leibler div.
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss(reduction="mean")(recon_x, x)

        # Kullback-Leibler divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        return recon_loss + kld_loss

    @classmethod
    def train_model(
        cls, dataset: str, batch_size: int, n_epochs: int, n_dataloader_workers: int
    ) -> nn.Module:
        """
        Train the variational autoencoder.
        """
        return _train(cls, dataset, batch_size, n_epochs, n_dataloader_workers)


def _train(
    model_cls: nn.Module,
    dataset: str,
    batch_size: int,
    n_epochs: int,
    n_dataloader_workers: int,
) -> nn.Module:
    """
    Train an autoencoder.
    """

    # Fetch the dataset
    dataset = ml_datasets.get_dataset(name=dataset, transform=model_cls.TRANSFORM)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True
    )

    # Instantiate the model
    model = model_cls(
        input_dim=model_cls.IMG_SIZE * model_cls.IMG_SIZE * 3, latent_dim=32
    )

    # Training mode & CUDA
    model.train()
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(n_epochs):
        print(f"+++ EPOCH {e+1}/{n_epochs} +++")
        total_loss = 0.0

        for batch, _ in tqdm(dataloader):
            # Flatten and move to device
            batch = batch.view(batch.size(0), -1).to(device)

            # Forward pass
            recon_batch, mu, logvar = model(batch)
            loss = model.loss(recon_batch, batch, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {e+1}/{n_epochs}: Loss {avg_loss:.4f}")
