 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import *
import pdb

def get_seq_model_shapes(seq_model, input_shape, seq_model_name = 'seq_model'):
    input_tensor = torch.zeros(*input_shape)
    output = input_tensor
    print("\n{} Layers:\n".format(seq_model_name))
    
    for i, ly in enumerate(seq_model):
        output = seq_model[i](output)
        print('Layer Block {}: {}, out shape: {}'.format(i, ly, output.shape))
    return output

def verify_string_args(string_arg, string_args_list):
    if string_arg not in string_args_list:
        raise ValueError("Argument '{}' not available in {}".format(string_arg, string_args_list))

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def convT_block(nf_in, nf_out, stride = 2, padding = 1,norm='no_norm', act=None, kernel_size=4):
    block = [nn.ConvTranspose2d(nf_in, nf_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True )]
    if act is not None:
        block.append(act)
    return block

def conv_block(nf_in, nf_out, stride = 2, padding = 2, fmap_shape=[10,10], norm=None, act=None, kernel_size=5):
    block = [nn.Conv2d(nf_in, nf_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True )]
    if norm == 'layer_norm':
        block.append(nn.LayerNorm([nf_out]+fmap_shape))
    elif norm == 'spectral_norm':
        block[-1] = torch.nn.utils.spectral_norm(block[-1])
    if act is not None:
        block.append(act)
    #block.append(nn.LeakyReLU(0.2, inplace=True))
    #block.append(GaussianNoise(normal_std_scale=0.7))
    return block

def linear_block(nf_in, nf_out, norm='no_norm',  act=None):
    block = [nn.Linear(nf_in, nf_out)]
    if norm == 'layer_norm':
        block.append(nn.LayerNorm([nf_out]))
    elif norm == 'spectral_norm':
        block[-1] = torch.nn.utils.spectral_norm(block[-1])
    if act is not None:
        block.append(act)
    return block

class Generator(nn.Module):
    def __init__(self, 
                opt,
                architecture = 'cnn', 
                nf=128, 
                kernel_size=4, 
                latent_dim = 100, 
                nc = 3,
                print_shapes=False,
                norm = 'no_norm'
                ):
        
        super(Generator, self).__init__()
        
        
        self.img_size = 28
        self.architecture = architecture
        self.nf = nf
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.nc = nc
        self.norm = norm
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        modules = nn.ModuleList()
        
        first_map_shape = 7
        # shared_layers = []
        # shared_layers += linear_block(self.latent_dim, nf*2*first_map_shape*first_map_shape, norm='no_norm', act=nn.ReLU(True))
        # shared_layers += Reshape(-1, nf*2, first_map_shape, first_map_shape),
        # shared_layers += convT_block(nf*2, nf, stride=2, padding=1, norm=self.norm, act=nn.ReLU(True)) 

        for _ in range(opt.n_paths_G):
            
            gen_layers = []

            if self.architecture == 'cnn' or self.architecture == 'cnn_short':
                # gen_layers += shared_layers
                gen_layers += linear_block(self.latent_dim, self.nf*2*first_map_shape*first_map_shape, norm='no_norm', act=nn.ReLU(True))
                gen_layers += Reshape(-1, self.nf*2, first_map_shape, first_map_shape),
                gen_layers += convT_block(self.nf*2, self.nf, stride=2, padding=1, norm=self.norm, act=nn.ReLU(True)) 
                gen_layers += convT_block(self.nf, opt.channels, stride=2, padding=1, norm='no_norm', act=nn.Tanh())  
            else:
                raise ValueError('Architecture {} not implemented!'.format(self.architecture))

            module = nn.Sequential(*gen_layers)
            modules.append(module)
            
        self.paths = modules

    def forward(self, z):
        img = []
        for path in self.paths:
            img.append(path(z).view(img.size(0), *self.opt.img_shape))
        img = torch.cat(img, dim=0)
       
        return img


# class Discriminator(nn.Module):
#     def __init__(self, opt):
#         super(Discriminator, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(4096, 100)
        
#         modules = nn.ModuleList()
        
#         modules.append(nn.Sequential(
#             nn.Linear(100, 1),
#             nn.Sigmoid(),
#                 ))
#         modules.append(nn.Sequential(
#             nn.Linear(100, opt.n_paths_G),
#                 ))
#         self.paths = modules
    
#     def forward(self, x):
        
#         x = F.leaky_relu(self.conv1(x), 0.2)
#         x = self.bn1(x)
#         x = F.leaky_relu(self.conv2(x), 0.2)
#         x = self.bn2(x)
#         x = F.leaky_relu(self.conv3(x), 0.2)
#         x = self.bn3(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
        
#         validity = self.paths[0](x)
#         classifier = F.log_softmax(self.paths[1](x), dim=1)

#         return validity, classifier



class Discriminator(nn.Module):
    def __init__(self, 
                 opt,
                 architecture='cnn',
                 nf=128, 
                 kernel_size=5, 
                 norm = 'layer_norm',
                 nc = 3,
                 print_shapes=True,
                 total_units_after_conv=100
                 ):
        super(Discriminator, self).__init__()

        self.img_size = 28
        self.architecture = architecture
        self.nf = nf
        self.kernel_size = kernel_size
        self.norm = norm
        self.nc = opt.channels
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        encoder_layers = []

        encoder_layers += conv_block(opt.channels, nf, fmap_shape=[14, 14], norm=self.norm, act=self.leaky_relu, kernel_size=self.kernel_size)
        encoder_layers += conv_block(nf, nf * 2, fmap_shape=[7, 7], norm=self.norm, act=self.leaky_relu, kernel_size=self.kernel_size) 
        encoder_layers += conv_block(nf * 2, nf * 4, fmap_shape=[4,4], norm=self.norm, act=self.leaky_relu, kernel_size=self.kernel_size) 

        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        modules = nn.ModuleList()
        
        total_units_after_conv = self.get_total_units_after_conv()
        
        modules.append(nn.Sequential(
            nn.Linear(total_units_after_conv, 1),
            nn.Sigmoid()
                ))
        modules.append(nn.Sequential(
            nn.Linear(total_units_after_conv, opt.n_paths_G),
                ))
        self.paths = modules

    def get_total_units_after_conv(self):
        input_tensor = torch.zeros(10, self.nc, self.img_size, self.img_size)
        output=input_tensor

        for i, ly in enumerate(self.encoder_layers):
            output = self.encoder_layers[i](output)

        total_units = output.view(input_tensor.shape[0], -1).shape[-1]

        return total_units
    
    def forward(self, x):
        
        # x = F.leaky_relu(self.conv1(x), 0.2)
        # x = self.bn1(x)
        # x = F.leaky_relu(self.conv2(x), 0.2)
        # x = self.bn2(x)
        # x = F.leaky_relu(self.conv3(x), 0.2)
        # x = self.bn3(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        
        x = self.encoder_layers(x)
        x = x.view(x.size(0), -1)
        validity = self.paths[0](x)
        classifier = F.log_softmax(self.paths[1](x), dim=1)

        return validity, classifier
        
