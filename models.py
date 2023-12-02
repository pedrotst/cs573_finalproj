import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import *

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.architecture = self.opt.architecture

        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        modules = nn.ModuleList()
        
        for _ in range(opt.n_paths_G):
            modules.append(
                nn.Sequential(
                *block(opt.latent_dim, 128),
                *block(128, 512),
                nn.Linear(512, int(np.prod(opt.img_shape))),
                nn.Tanh()
            ))
        self.paths = modules

    def forward(self, z):
        img = []
        for path in self.paths:
            img.append(path(z).view(img.size(0), *self.opt.img_shape))
        img = torch.cat(img, dim=0)
        return img

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(int(np.prod(opt.img_shape)), 512)
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)
        
        encoder_layers = []

        encoder_layers += [self.fc1]
        encoder_layers += [self.lr1]
        encoder_layers += [self.fc2]
        encoder_layers += [self.lr2]
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        modules = nn.ModuleList()
        
        modules.append(nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
                ))
        modules.append(nn.Sequential(
            nn.Linear(256, opt.n_paths_G),
                ))
        self.paths = modules
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        # img_flat = self.lr2(self.fc2(self.lr1(self.fc1(img_flat))))
        img_flat = self.encoder_layers(img_flat)
        
        validity = self.paths[0](img_flat)
        classifier = F.log_softmax(self.paths[1](img_flat), dim=1)
        
        return validity, classifier
