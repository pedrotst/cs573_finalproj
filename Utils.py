import matplotlib.pyplot as plt
import numpy as np

class arg_parser_subst():
    
    def __init__(self, argv):

        if(len(argv) == 15):
            self.n_epochs = int(argv[1])
            self.batch_size = int(argv[2])
            self.batch_size_g = int(argv[3])
            self.lr = int(argv[4])
            self.b1 = int(argv[5])
            self.b2 = int(argv[6])
            self.n_cpu = int(argv[7])
            self.latent_dim = int(argv[8])
            self.img_size = int(argv[9])
            self.channels = int(argv[10])
            self.sample_interval = int(argv[11])
            self.n_paths_G = int(argv[12])
            self.classifier_para = int(argv[13])
            self.classifier_para_g = int(argv[14])
            self.img_shape = (self.channels, self.img_size, self.img_size)
        else:
            self.n_epochs = 500 
            self.batch_size = 100
            self.batch_size_g = 100
            self.lr = 0.0002 
            self.b1 = 0.5 
            self.b2 = 0.999 
            self.n_cpu = 12 
            self.latent_dim = 100 
            self.img_size = 28
            self.channels = 1
            self.sample_interval = 400 
            self.n_paths_G = 50 # number of generators
            self.classifier_para = 0.001
            self.classifier_para_g = 0.001
            self.img_shape = (self.channels, self.img_size, self.img_size)

def show(img, opt):
    npimg = img.detach().numpy()
    plt.figure(figsize = (opt.n_paths_G//2,opt.n_paths_G//2))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
