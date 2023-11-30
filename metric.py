import torch
import torch.nn.functional as F

def cross_entropy(a,b):
    a_normalized = F.softmax(a, dim=0)
    b_normalized = F.softmax(b, dim=0)

    return cross_entropy = -torch.sum(a_normalized * torch.log(b_normalized))