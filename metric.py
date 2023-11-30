import torch
import torch.nn.functional as F

def cross_entropy(a,b):
    a_normalized = F.softmax(a, dim=0)
    b_normalized = F.softmax(b, dim=0)

    return cross_entropy = -torch.sum(a_normalized * torch.log(b_normalized))

def normalize_distribution(p):
    """
    Normalize a probability distribution.
    """
    return p / np.sum(p)

def kl_divergence(p, q):
    """
    Kullback-Leibler Divergence
    """
    p_normalized = normalize_distribution(p)
    q_normalized = normalize_distribution(q)
    return np.sum(np.where(p_normalized != 0, p_normalized * np.log(p_normalized / q_normalized), 0))

def js_divergence(p, q):
    """
    Jensen-Shannon Divergence
    """
    p_normalized = normalize_distribution(p)
    q_normalized = normalize_distribution(q)
    m = 0.5 * (p_normalized + q_normalized)
    return 0.5 * kl_divergence(p_normalized, m) + 0.5 * kl_divergence(q_normalized, m)

def cross_entropy(p, q):
    """
    Cross Entropy
    """
    p_normalized = normalize_distribution(p)
    q_normalized = normalize_distribution(q)
    return -np.sum(p_normalized * np.log(q_normalized))

def total_variation_distance(p, q):
    """
    Total Variation Distance
    """
    p_normalized = normalize_distribution(p)
    q_normalized = normalize_distribution(q)
    return 0.5 * np.sum(np.abs(p_normalized - q_normalized))