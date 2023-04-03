'''
reference
- https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
- https://velog.io/@hong_journey/VAEVariational-AutoEncoder-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0
- https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(
        self,
        x_dim,
        h_dim1,
        h_dim2,
        z_dim,
        drop_prob
    ):
        super(VAE, self).__init__()

        ### encoder ###
        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim, h_dim1),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )

        # 2nd hidden layer
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )

        # output layer
        self.mu = nn.Linear(h_dim2, z_dim)
        self.log_var = nn.Linear(h_dim2, z_dim)

        ### decoder ###
        # 1st hidden layer
        self.fc4 = nn.Sequential(
            nn.Linear(z_dim, h_dim2),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )

        # 2nd hidden layer
        self.fc5 = nn.Sequential(
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )

        # output layer
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def reparametrize(mu, log_var):
        std = torch.exp(0.5 *log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encoder(self, x):
        x = self.fc2(self.fc1(x))

        mu = F.relu(self.mu(x))
        log_var = F.relu(self.log_var(x))

        z = self.reparametrize(mu, log_var)

        return z, mu, log_var

    def decoder(self, z):
        z = self.fc5(self.fc4(z))
        reconst_x = F.sigmoid(self.fc6(z))

        return reconst_x
    
    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        reconst_x = self.decoder(z)

        return reconst_x, mu, log_var
    
# reconstruction loss + KL divergence
def vae_loss(x, reconst_x, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(reconst_x, x, reduce='sum')
    kld = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)

    loss = reconstruction_loss + kld

    return loss