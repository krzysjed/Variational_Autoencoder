import torch
import torch.utils.data
from torch import nn
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def softclip(tensor, minimum):
    return minimum + F.softplus(tensor - minimum)


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=(3, 3), padding=1))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out


def image_plot(recon, img):
    # save_image(recon, 'input' + '.png')
    # save_image(img, 'rec' + '.png')
    recon = make_grid(recon).permute(1, 2, 0).detach().numpy()
    img = make_grid(img).permute(1, 2, 0).detach().numpy()

    plt.ion()

    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.axis(False)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.axis(False)
    plt.imshow(recon)
    plt.waitforbuttonpress()
    plt.show()


class VAE(LightningModule):
    def __init__(self, start_beta=1, max_beta=1):
        super(VAE, self).__init__()

        self.rec = 0
        self.z_dim = 64
        self.log_sigma = 0

        self.max_beta = max_beta
        self.beta = start_beta

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),

            ResidualBlock(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),

            ResidualBlock(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 128, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 128, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=(9, 9), stride=(2, 2)),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=(12, 12), stride=(1, 1)),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Sigmoid(),

        )

        # fully connected layers for learning representations

        self.fc_mu = nn.Linear(128 * 4 * 4, self.z_dim)
        self.fc_var = nn.Linear(128 * 4 * 4, self.z_dim)
        self.fc2 = nn.Linear(self.z_dim, 128 * 4 * 4)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)

        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

    def decode(self, x):

        x = self.fc2(x)
        x = x.view(-1, 128, 4, 4)

        out = self.decoder(x)
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def step(self, x):
        x, y = x

        x_hat, kl = self._run_step(x)

        rec = self.reconstruction_loss(x, x_hat)

        total_loss = rec + (kl * self.beta)

        logs = {
            "beta": self.beta,
            "rec_loss": rec / len(x),
            "kl_loss": kl / len(x),
            "total_loss": total_loss / len(x),
            "log_sigma": self.log_sigma
        }

        return total_loss, logs

    def _run_step(self, x):
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return self.decode(z), kld

    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        sample = self.fc2(sample)
        sample = sample.view(-1, 128, 4, 4)
        return self.decoder(sample)

    def reconstruction_loss(self, x_hat, x):
        log_sigma = ((x - x_hat) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
        self.log_sigma = log_sigma.item()

        log_sigma = softclip(log_sigma, -6)

        rec = gaussian_nll(x_hat, log_sigma, x).sum()
        self.rec = rec
        return rec

    def training_step(self, package, batch_idx):
        loss, logs = self.step(package)
        self.log_dict({f"Epoch_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def on_epoch_end(self) -> None:
        if self.beta == self.max_beta:
            pass
        elif self.rec >= 0 and self.beta > 0.2:
            self.beta -= 0.01
        elif self.rec < 0:
            self.beta += round(self.max_beta / 1000)

        if self.beta > self.max_beta:
            self.beta = self.max_beta

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        sheduler = torch.optim.lr_scheduler.StepLR(optim, 2000, 0.7)
        return {"optimizer": optim, "lr_scheduler": sheduler}

    def test_step(self, x, xz):
        x, y = x
        x_hat, kl = self._run_step(x)
        image_plot(x_hat, x)
