'''
Reference
- paper link: https://arxiv.org/pdf/1312.6114.pdf
- code link: https://github.com/pytorch/examples/blob/main/vae/main.py
'''

import argparse
import os
import random
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument(
    '--batch_size', 
    type=int, 
    default=128,
    help='input batch size (default: 128)'      
)
parser.add_argument(
    '--epochs',
    type=int,
    default=20,
    help='number of epochs to train (default: 20)'
)
# parser.add_argument(
#     '--x_dim',
#     type=int,
#     default=784,
#     help='input dimension (default: 784)'
# )
parser.add_argument(
    '--img_size',
    type=int,
    default=28,
    help='input image size (default: 28)'
)
parser.add_argument(
    '--h_dim',
    type=int,
    default=400,
    help='hidden layer dimension (default: 400)'
)
parser.add_argument(
    '--z_dim',
    type=int,
    default=20,
    help='latent vector dimension (default: 20)'
)
parser.add_argument(
    '--seed',
    type=int,
    default=2023,
    help='random seed (default: 2023)'
)
args = parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class VAE(nn.Module):
    def __init__(
        self,
        x_dim,
        h_dim,
        z_dim
    ):
        super(VAE, self).__init__()

        self.x_dim = x_dim

        ### encoder ###
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.mu = nn.Linear(h_dim, z_dim)
        self.log_var = nn.Linear(h_dim, z_dim)

        ### decoder ###
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encode(self, x):
        h = F.relu(self.fc1(x))

        mu = self.mu(h)
        log_var = self.log_var(h)

        z = self.reparametrize(mu, log_var)

        return z, mu, log_var
        
    def decode(self, z):
        h = F.relu(self.fc3(z))
        reconst_x = self.fc4(h)

        return torch.sigmoid(reconst_x)
        
    def forward(self, x):
        z, mu, log_var = self.encode(x.view(-1, self.x_dim))
        reconst_x = self.decode(z)

        return reconst_x, mu, log_var

# reconstruction loss + KL divergence regularizatioin        
def loss_function(reconst_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(reconst_x, x.view(-1, x.size(-1) ** 2), reduction='sum')
    # Appendix B from VAE paper
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = BCE + KLD

    return loss

def train(epoch, model, train_loader, optimizer):
    model.train()
    train_loss = 0

    for data, _ in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconst_batch, mu, log_var = model(data)
        loss = loss_function(reconst_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    _train_loss = train_loss / len(train_loader.dataset)

    print(f'Epoch: {epoch} Train Loss: {_train_loss:.4f}')

def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            reconst_batch, mu, log_var = model(data)
            loss = loss_function(reconst_batch, data, mu, log_var)
            test_loss += loss.item()

            if batch_idx == 0 and epoch % 10 == 0:
                batch_size = data.size(0)
                n = min(batch_size, 8)
                comparison = torch.cat([data[:n], reconst_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(
                    comparison.cpu(),
                    f'./results/reconstruction_{epoch}.png',
                    nrow=n
                )

        _test_loss = test_loss / len(test_loader.dataset)
        print(f'Test Loss: {_test_loss:.4f}')

if __name__ == '__main__':
    seed_everything(args.seed)

    ### set device ###
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps:0')
    else:
        device = torch.device('cpu')
    print(device)

    ### set directory ###
    if not os.path.isdir('./results/'):
        os.mkdir('./results/')

    ### load model ###
    vae = VAE(
        # args.x_dim,
        args.img_size ** 2,
        args.h_dim,
        args.z_dim
    ).to(device)
    print(vae)

    ### dataloader ###
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=False
    )

    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, vae, train_loader, optimizer)
        test(epoch, vae, test_loader)

        if epoch % 10 == 0:
            with torch.no_grad():
                sample = torch.randn(64, args.z_dim).to(device)
                sample = vae.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28), f'./results/sample_{epoch}.png')

