import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model.dcgan import Generator, Discriminator


__all__ = [ "MNISTAgent", "CIFAR10Agent" ]

class MNISTAgent:

    def __init__(self, config):
        ### Configuration option
        self.config = config

        ### Training dataset
        transform = transforms.Compose([
                                transforms.Resize(config['dataset']['input_size']),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root="download/",
                                transform=transform,
                                download=True,
                                train=True)
        self.dataloader = DataLoader(dataset,
                                batch_size=config['dataset']['batch_size'],
                                shuffle=True)

        ### Training environment
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### Models to train
        self.netD = Discriminator(channels_img=config['discriminator']['channels_img'],
                                features_d=config['discriminator']['features_d']).to(self.device)
        self.netG = Generator(channels_noise=config['generator']['channels_noise'],
                            channels_img=config['generator']['channels_img'],
                            features_g=config['generator']['features_g']).to(self.device)

        ### Optimzer & Loss function
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        ### Validation & Tensorboard
        self.fixed_noise = torch.randn(64, config['generator']['channels_noise'], 1, 1).to(self.device)
        self.writer = SummaryWriter(osp.join(config['tensorboard']['dir'], 'dcgan'))
        self.writer_real = SummaryWriter(osp.join(config['tensorboard']['dir'], 'real'))
        self.writer_fake = SummaryWriter(osp.join(config['tensorboard']['dir'], 'fake'))

    def train(self):
        for epoch in range(self.config['train']['n_epochs']):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        for batch_idx, (data, target) in enumerate(self.dataloader):

            data = data.to(self.device)
            batch_size = data.shape[0]

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            self.netD.zero_grad()
            label = (torch.ones(batch_size)*0.9).to(self.device)
            output = self.netD(data).reshape(-1)
            lossD_real = self.criterion(output, label)
            D_x = output.mean().item()

            noise = torch.randn(batch_size, self.config['generator']['channels_noise'], 1, 1).to(self.device)
            fake = self.netG(noise)
            label = (torch.ones(batch_size)*0.1).to(self.device)
            output = self.netD(fake.detach()).reshape(-1)
            lossD_fake = self.criterion(output, label)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            self.optimizerD.step()

            ### Train Generator: max log(D(G(z))
            self.netG.zero_grad()
            label = torch.ones(batch_size).to(self.device)
            output = self.netD(fake).reshape(-1)
            lossG = self.criterion(output, label)
            lossG.backward()
            self.optimizerG.step()

            if batch_idx % 100 == 0:
                print(("Epoch [{}:{}] Progress: [{}:{}], Discriminator Loss: {:.2f}, Generator Loss {:.2f}, "
                        "D(X): {:.2f}").format(
                        epoch, self.config['train']['n_epochs'], batch_idx,
                        len(self.dataloader), lossD.item(), lossG.item(), D_x))

                with torch.no_grad():
                    fake = self.netG(self.fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    self.writer_real.add_image('MNIST Real Iamges', img_grid_real, batch_idx+epoch*len(self.dataloader))
                    self.writer_fake.add_image('MNIST Fake Iamges', img_grid_fake, batch_idx+epoch*len(self.dataloader))

                self.writer.add_scalar('Discriminator Loss', lossD.item(), batch_idx+epoch*len(self.dataloader))
                self.writer.add_scalar('Generator Loss', lossG.item(), batch_idx+epoch*len(self.dataloader))

    def validate(self):
        pass

    def finalize(self):
        pass

    def save_checkpoint(self):
        pass

    def import_checkpoint(self):
        pass


class CIFAR10Agent:

    def __init__(self, config):
        ### Configuration option
        self.config = config

        ### Training dataset
        transform = transforms.Compose([
                                transforms.Resize(config['dataset']['input_size']),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.CIFAR10(root="download/",
                                transform=transform,
                                download=True,
                                train=True)
        self.dataloader = DataLoader(dataset,
                                batch_size=config['dataset']['batch_size'],
                                shuffle=True)

        ### Training environment
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### Models to train
        self.netD = Discriminator(channels_img=config['discriminator']['channels_img'],
                                features_d=config['discriminator']['features_d']).to(self.device)
        self.netG = Generator(channels_noise=config['generator']['channels_noise'],
                            channels_img=config['generator']['channels_img'],
                            features_g=config['generator']['features_g']).to(self.device)

        ### Optimzer & Loss function
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        ### Validation & Tensorboard
        self.fixed_noise = torch.randn(64, config['generator']['channels_noise'], 1, 1).to(self.device)
        self.writer = SummaryWriter(osp.join(config['tensorboard']['dir'], 'dcgan'))
        self.writer_real = SummaryWriter(osp.join(config['tensorboard']['dir'], 'real'))
        self.writer_fake = SummaryWriter(osp.join(config['tensorboard']['dir'], 'fake'))

    def train(self):
        for epoch in range(self.config['train']['n_epochs']):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        for batch_idx, (data, target) in enumerate(self.dataloader):

            data = data.to(self.device)
            batch_size = data.shape[0]

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            self.netD.zero_grad()
            label = (torch.ones(batch_size)*0.9).to(self.device)
            output = self.netD(data).reshape(-1)
            lossD_real = self.criterion(output, label)
            D_x = output.mean().item()

            noise = torch.randn(batch_size, self.config['generator']['channels_noise'], 1, 1).to(self.device)
            fake = self.netG(noise)
            label = (torch.ones(batch_size)*0.1).to(self.device)
            output = self.netD(fake.detach()).reshape(-1)
            lossD_fake = self.criterion(output, label)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            self.optimizerD.step()

            ### Train Generator: max log(D(G(z))
            self.netG.zero_grad()
            label = torch.ones(batch_size).to(self.device)
            output = self.netD(fake).reshape(-1)
            lossG = self.criterion(output, label)
            lossG.backward()
            self.optimizerG.step()

            if batch_idx % 100 == 0:
                print(("Epoch [{}:{}] Progress: [{}:{}], Discriminator Loss: {:.2f}, Generator Loss {:.2f}, "
                        "D(X): {:.2f}").format(
                        epoch, self.config['train']['n_epochs'], batch_idx,
                        len(self.dataloader), lossD.item(), lossG.item(), D_x))

                with torch.no_grad():
                    fake = self.netG(self.fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    self.writer_real.add_image('CIFAR Real Images', img_grid_real, batch_idx+epoch*len(self.dataloader))
                    self.writer_fake.add_image('CIFAR Fake Images', img_grid_fake, batch_idx+epoch*len(self.dataloader))

                self.writer.add_scalar('Discriminator Loss', lossD.item(), batch_idx+epoch*len(self.dataloader))
                self.writer.add_scalar('Generator Loss', lossG.item(), batch_idx+epoch*len(self.dataloader))

    def validate(self):
        pass

    def finalize(self):
        pass

    def save_checkpoint(self):
        pass

    def import_checkpoint(self):
        pass
