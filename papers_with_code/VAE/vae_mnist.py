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
import torchvision
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
parser.add_argument(
    '--data',
    type=str,
    default='MNIST',
    help='dataset option: MNIST, CIFAR-10 (default: MNIST)'
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
    BCE = F.binary_cross_entropy(reconst_x, x.view(reconst_x.size()), reduction='sum')
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
            if batch_idx == 0:
                _, input_channel, input_size, _ = data.size()

            data = data.to(device)
            reconst_batch, mu, log_var = model(data)
            loss = loss_function(reconst_batch, data, mu, log_var)
            test_loss += loss.item()

            if batch_idx == 0 and epoch % 10 == 0:
                batch_size = data.size(0)
                n = min(batch_size, 8)
                comparison = torch.cat([data[:n], reconst_batch.view(batch_size, input_channel, input_size, input_size)[:n]])
                save_image(
                    comparison.cpu(),
                    f'./results/{args.data}/reconstruction_{epoch}.png',
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
    results_dir = f'./results/{args.data}/'

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    ### dataloader ###
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.data == 'MNIST':
        train_dataset = datasets.MNIST(
            '../data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )

        test_dataset = datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.ToTensor()
        )
    elif args.data == 'CIFAR-10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='../data', 
            train=True, 
            download=True, 
            transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=False, 
            download=True, 
            transform=transform_test
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    input_channel, input_size, _ = train_dataset[0][0].size()
    x_dim = input_channel * (input_size ** 2)

    ### load model ###
    vae = VAE(
        x_dim,
        args.h_dim,
        args.z_dim
    ).to(device)
    print(vae)

    optimizer = optim.Adam(vae.parameters(), lr=1e-5)

    for epoch in range(1, args.epochs + 1):
        train(epoch, vae, train_loader, optimizer)
        test(epoch, vae, test_loader)

        if epoch % 10 == 0:
            with torch.no_grad():
                sample = torch.randn(64, args.z_dim).to(device)
                sample = vae.decode(sample).cpu()
                save_image(sample.view(64, input_channel, input_size, input_size), f'./results/{args.data}/sample_{epoch}.png')

