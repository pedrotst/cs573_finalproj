import matplotlib.pyplot as plt
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--batch_size_g', type=int, default=100, help='generator batch size')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum')
    parser.add_argument('--n_cpu', type=int, default=12, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--sample_interval', type=int, default=1, help='interval between image samples')
    parser.add_argument('--classifier_para', type=float, default=1.0, help='classifier loss weight')
    parser.add_argument('--classifier_para_g', type=float, default=1.0, help="generator's classification loss weight")
    parser.add_argument('--dataset', choices=['fmnist', 'mnist', 'cifar10'], help='dataset name')
    parser.add_argument('--architecture', default='mlp', help='architecture for the model')
    parser.add_argument('--logs_path', default='logs', help='where to save logs')
    parser.add_argument('--init_noise', type=float, default=1.0, help='Initial stdd of gaussian noise applied to input of discriminator')
    parser.add_argument('--noise_decay', type=float, default=0.01, help='How much is subtracted of of gaussian stdd noise at each epoch')
    
    #generator parameters
    parser.add_argument('--n_paths_G', type=int, default=50, help='number of generators')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate for gen')
    
    parser.add_argument('--nf_g', type=int, default=128,help='Number of feature maps for generator.')
    # parser.add_argument('--norm_g', type=str, default='layer_norm', choices=['layer_norm', 'batch_norm', 'no_norm'], help='Type of normalization layer used for generator')
    parser.add_argument('--no_conv_g', type=int, default=2, choices=[2], help='Number of conv layers for gen')
    
    #discriminator/classifier parameters
    parser.add_argument('--lr_d', type=float, default=0.0002, help='learning rate for disc')
    parser.add_argument('--lr_c', type=float, default=0.00002, help='learning rate for clasf')
    parser.add_argument('--nf_d', type=int, default=128, help='Number of feature maps for discriminator.')
    parser.add_argument('--norm_d', type=str, default='layer_norm', choices=['layer_norm', 'batch_norm', 'no_norm'], help='Type of normalization layer used for discriminator')
    parser.add_argument('--no_conv_d', type=int, default=3, choices=[3], help='Number of conv layers for discriminator')
    
    
    parser.set_defaults(shared_features_across_ref=False)


    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.img_size = 32
        args.channels = 3
    else:  
        args.img_size = 28
        args.channels = 1

    args.img_shape = (args.channels, args.img_size, args.img_size)

    return args


def show(img, opt):
    npimg = img.detach().numpy()
    dim = max(5, opt.n_paths_G//2)
    plt.figure(figsize = (dim, dim))
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


def get_cluster(dataloader, discriminator, soft_preds=False, noise_intensity=0):
    #cuda = True if torch.cuda.is_available() else False
    #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_data = []
    all_idxs = []
    all_preds = []
    with torch.no_grad():
        for i, (x,y) in enumerate(dataloader):
            x = x.to(device)
            x = add_noise(x, noise_intensity)
            y = y.to(device)
            d_out, c_out = discriminator(x)
            all_preds.append(c_out)
            all_idxs.append(y)
    all_idxs = torch.cat(all_idxs, axis=0)
    all_preds = torch.cat(all_preds, axis=0)
    if soft_preds:
        return all_idxs, all_preds
    else:
        return all_idxs, torch.argmax(all_preds, dim=-1)



from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    

def add_noise(tensor, normal_std_scale):
    if (normal_std_scale > 0):
        return tensor + (tensor*torch.randn_like(tensor)*normal_std_scale)
    else:
        return tensor