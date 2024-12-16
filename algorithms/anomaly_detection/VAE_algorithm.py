import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
import pytorch_lightning as pl
import logging

from abc import ABC, abstractmethod
import pandas as pd

class AnomalyDetectionAlgorithm(ABC):
    @abstractmethod
    def detect_anomalies(self, df,dataset):
        """
        Detect anomalies in the given dataframe.
        
        :param df: DataFrame containing the time series data
        :return: DataFrame with anomalies detected (e.g., with anomaly scores)
        """
        pass


class VAEAnomalyDetection(pl.LightningModule):
    """
    Variational Autoencoder (VAE) for anomaly detection using PyTorch Lightning.
    """

    def __init__(self, input_size: int, latent_size: int, L: int = 10, lr: float = 1e-3, log_steps: int = 1000):
        super().__init__()
        self.L = L
        self.lr = lr
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = self.make_encoder(input_size, latent_size)
        self.decoder = self.make_decoder(latent_size, input_size)
        self.prior = Normal(0, 1)
        self.log_steps = log_steps

    def make_encoder(self, input_size: int, latent_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size * 2)
        )

    def make_decoder(self, latent_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size * 2)
        )

    def forward(self, x: torch.Tensor) -> dict:
        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # Broadcast input across sample dimension (L)
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(dim=0)
        log_lik = log_lik.mean(dim=0).sum()
        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik
        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

    def predict(self, x) -> dict:
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1)
        latent_sigma = softplus(latent_sigma)
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.L])
        z = z.view(self.L * batch_size, self.latent_size)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        recon_mu = recon_mu.view(self.L, *x.shape)
        recon_sigma = recon_sigma.view(self.L, *x.shape)
        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def is_anomaly(self, x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        p = self.reconstructed_probability(x)
        return p < alpha

    def reconstructed_probability(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = self.predict(x)
        recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])
        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)
        return p

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)

    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        self.log('train/loss', loss['loss'])
        self.log('train/kl_loss', loss['kl'], prog_bar=False)
        self.log('train/recon_loss', loss['recon_loss'], prog_bar=False)
        return loss['loss']

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        self.log('val/loss', loss['loss'], on_epoch=True)
        self.log('val/kl_loss', loss['kl'], on_epoch=True)
        self.log('val/recon_loss', loss['recon_loss'], on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class VAEAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.feature = 'meantemp'
        self.latent_dim = 2
        self.threshold = None
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=logging.INFO
        )
        self.logger.info(f"{self.__class__.__name__} class instantiated")

    def process_data(self, dataset):
        data = dataset[[self.feature]].values.astype('float32')
        return data

    def train_model(self, data):
        input_dim = data.shape[1]
        dataset = torch.tensor(data)
        self.model = VAEAnomalyDetection(input_size=input_dim, latent_size=self.latent_dim)
        trainer = pl.Trainer(max_epochs=50, logger=False, enable_checkpointing=False)
        trainer.fit(self.model, torch.utils.data.DataLoader(dataset, batch_size=32))

        # Compute reconstruction probability on training data
        with torch.no_grad():
            p = self.model.reconstructed_probability(dataset)
        self.threshold = torch.quantile(p, 0.05).item()
        self.logger.info(f"Anomaly detection threshold set at: {self.threshold}")

    def detect_anomalies(self, df, dataset):
        self.logger.info(f"{self.__class__.__name__} - detect_anomalies method invoked")

        try:
            data = self.process_data(dataset)
            if np.isnan(data).any() or np.isinf(data).any():
                self.logger.error("Training data contains NaNs or infinite values")
                raise ValueError("Training data contains NaNs or infinite values.")

            if self.model is None:
                self.train_model(data)

            new_data = self.process_data(df)
            if np.isnan(new_data).any() or np.isinf(new_data).any():
                self.logger.error("New data contains NaNs or infinite values")
                raise ValueError("New data contains NaNs or infinite values.")

            new_dataset = torch.tensor(new_data)
            with torch.no_grad():
                is_anomaly = self.model.is_anomaly(new_dataset, alpha=self.threshold)

            anomalies = df[is_anomaly.numpy()]
            anomalies.reset_index(drop=True, inplace=True)
            self.logger.info(f"Anomalies detected: {len(anomalies)}")

            return anomalies

        except Exception as e:
            self.logger.error(f"Error in detect_anomalies: {e}")
            import traceback
            traceback.print_exc()
            raise e
