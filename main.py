import os
import sys

import utils
# import models
import torch


from tqdm import tqdm_notebook
from torch.utils.data import DataLoader
from torchvision import datasets
from gan import *
import torchvision.transforms as transforms
import torch.nn as nn

def main():
    opt = utils.arg_parser_subst(sys.argv)

    if("cnn" in opt.architecture):
        import models_cnn as models
    else:
        import models

    generator = models.Generator(opt) #ok 
    discriminator = models.Discriminator(opt) # ok

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    gan = Gan(generator, discriminator, optimizer_G, optimizer_D, opt)

    # Configure data loader
    os.makedirs("data/fmnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data/fmnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    gan.train(dataloader)


if __name__ == "__main__":
    main()
