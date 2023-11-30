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
    # opt = utils.arg_parser_subst(sys.argv)
    opt = utils.parse_args()

    if opt.dataset == "fmnist":
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
        if("cnn" in opt.architecture):
            import models_cnn_28 as models
        else:
            import models
        


    if opt.dataset == "mnist":
        os.makedirs("data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
        if("cnn" in opt.architecture):
            import models_cnn_28 as models
        else:
            import models
        
        
        
    elif opt.dataset == "cifar10":
        os.makedirs("data/CIFAR10", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/CIFAR10', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([.5],[.5])
                   ])
            ),
            batch_size=opt.batch_size, 
            shuffle=True)
        if("cnn" in opt.architecture):
            import models_cnn_32 as models
        else:
            import models

    generator = models.Generator(opt) #ok 
    discriminator = models.Discriminator(opt) # ok

    #optimizer G
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    
    #optimizer D
    optimizer_D = torch.optim.Adam(list(discriminator.encoder_layers.parameters()) + list(discriminator.paths[0].parameters()), lr=opt.lr_d, betas=(opt.b1, opt.b2))
    # optimizer_D.add_param_group(discriminator.paths[0].parameters())
    
    #optimizer C
    optimizer_C = torch.optim.Adam(discriminator.paths[1].parameters(), lr=opt.lr_c, betas=(opt.b1, opt.b2))
    gan = Gan(generator, discriminator, optimizer_G, optimizer_D, optimizer_C, opt)

    gan.train(dataloader)


if __name__ == "__main__":
    main()
