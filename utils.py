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


    
# -----------------------------------------------------
#  Metrics Computation
# -----------------------------------------------------
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import numpy as np
import torch


def clustering_acc(y_true, y_pred):
    """
    Calculate the clustering accuracy for the case where the number of clusters
    might be different from the number of true classes. It uses the Hungarian algorithm.

    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted cluster labels.

    Returns:
    float: Clustering accuracy.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row, col] for row, col in zip(row_ind, col_ind)]) / y_pred.size


def get_cluster(dataloader, discriminator, soft_preds=False):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    all_data = []
    all_idxs = []
    all_preds = []
    for i, (x,y) in enumerate(dataloader):
        x = x.type(Tensor)
        y = y.type(Tensor)
        d_out, c_out = discriminator(x)
        all_preds.append(c_out)
        all_idxs.append(y)
    all_idxs = torch.cat(all_idxs, axis=0)
    all_preds = torch.cat(all_preds, axis=0)
    if soft_preds:
        return all_idxs, all_preds
    else:
        return all_idxs, torch.argmax(all_preds, dim=-1)