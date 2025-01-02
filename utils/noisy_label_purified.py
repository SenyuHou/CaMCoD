import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

def knn_cos(query, data, k=50, use_cosine_similarity=False):
    assert data.shape[1] == query.shape[1]

    if use_cosine_similarity:
        # Normalize feature vectors for cosine similarity calculation
        query_norm = query / query.norm(dim=1)[:, None]
        data_norm = data / data.norm(dim=1)[:, None]
        # Calculate cosine similarity
        sim = torch.mm(query_norm, data_norm.t())
        # Select top k highest similarities
        v, ind = sim.topk(k, largest=True)
    else:
        # Calculate Euclidean distance
        M = torch.cdist(query, data)
        # Select top k smallest distances
        v, ind = M.topk(k, largest=False)

    return v, ind[:, 0:min(k, data.shape[0])].to(torch.long)

def label_distribution(query_embd, y_query, prior_embd, labels, k=50, n_class=10, weighted=True, use_cosine_similarity=True):
    n_sample = query_embd.shape[0]
    device = query_embd.device

    # Get nearest neighbor indices and weights
    neighbour_v, neighbour_ind = knn_cos(query_embd, prior_embd, k=k, use_cosine_similarity=use_cosine_similarity)

    # Initialize label distribution
    neighbour_label_distribution = torch.zeros((n_sample, n_class), device=device)

    # Get neighbor labels
    neighbour_labels = labels[neighbour_ind]

    if weighted:
        # Compute weights, smaller distance gets larger weight
        weights = 1.0 / (neighbour_v + 1e-6)  # Add small value to avoid division by zero
        weights_normalized = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights

        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(neighbour_labels, num_classes=n_class).float()  # [n_sample, k, n_class]

        # Update label distribution using weights
        neighbour_label_distribution = torch.sum(labels_one_hot * weights_normalized.unsqueeze(2), dim=1)

    else:
        # For unweighted case, still use one-hot encoding but calculate mean
        labels_one_hot = F.one_hot(neighbour_labels, num_classes=n_class).float()  # [n_sample, k, n_class]
        neighbour_label_distribution = labels_one_hot.mean(dim=1)  # Calculate mean instead of weighted mean

    # Find the labels with the highest probability in the label distribution
    _, max_prob_label = torch.max(neighbour_label_distribution, dim=1)

    return max_prob_label, neighbour_label_distribution

def KL_label_distribution(neighbour_label_distribution_w, neighbour_label_distribution_s):
    # Ensure the distributions are properly normalized
    distribution_w = F.softmax(neighbour_label_distribution_w, dim=1)
    distribution_s = F.softmax(neighbour_label_distribution_s, dim=1)

    # Compute KL divergence per sample
    kl_div = F.kl_div(distribution_w.log(), distribution_s, reduction='none')

    # Sum over the class dimension to get KL divergence per sample
    kl_div_per_sample = kl_div.sum(dim=1)

    return kl_div_per_sample

def gmm_binary_split(kl_div_values, n_components=2, random_state=0):
    # Ensure kl_div_values is a NumPy array
    if isinstance(kl_div_values, torch.Tensor):
        kl_div_values = kl_div_values.numpy()

    # Reshape for GMM
    kl_div_values = kl_div_values.reshape(-1, 1)

    # Train GMM
    gmm = GaussianMixture(n_components=n_components, random_state=random_state).fit(kl_div_values)

    # Get means of the two components and determine the lower mean
    means = gmm.means_.flatten()
    lower_mean_idx = np.argmin(means)

    # Predict component labels for each KL divergence value
    component_labels = gmm.predict(kl_div_values)

    # Assign samples to lower or higher set based on component labels
    lower_set_batch = np.where(component_labels == lower_mean_idx)[0]
    higher_set_batch = np.where(component_labels != lower_mean_idx)[0]

    return lower_set_batch, higher_set_batch

def sample_labels(neighbour_label_distribution, y_query, max_prob_label, lower_set_batch, higher_set_batch, to_single_label=False):
    y_label_batch = torch.zeros_like(neighbour_label_distribution)

    # For samples in higher set, retain the probability distribution as labels
    y_label_batch[higher_set_batch] = neighbour_label_distribution[higher_set_batch]

    # For samples in lower set, check if the original label matches the max probability label
    for idx in lower_set_batch:
        if y_query[idx] == max_prob_label[idx]:
            # If original label matches max probability label, retain original label in one-hot format
            y_label_batch[idx] = F.one_hot(y_query[idx], num_classes=neighbour_label_distribution.shape[1]).float()
        else:
            # Otherwise, use the max probability label in one-hot format
            y_label_batch[idx] = F.one_hot(max_prob_label[idx], num_classes=neighbour_label_distribution.shape[1]).float()

    if to_single_label:
        # If to_single_label is True, convert to single integer labels
        y_label_batch = torch.argmax(y_label_batch, dim=1)

    return y_label_batch

def get_loss_weights(query_embd, y_query, prior_embd, labels, k=10, n_class=10):
    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn_cos(query_embd, prior_embd, k=k, use_cosine_similarity=False)

    # Compute the labels of the nearest neighbors
    neighbour_label_distribution = labels[neighbour_ind]

    # Append the label of the query
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)

    # Sample a label from the k+1 labels (k neighbors and itself)
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), torch.randint(0, k+1, (n_sample,))]

    # Convert labels to bincount (row wise)
    y_one_hot_batch = F.one_hot(neighbour_label_distribution, num_classes=n_class).float()

    # Compute the frequency of the sampled labels
    neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), sampled_labels]

    # Normalize max count as weight
    weights = neighbour_freq / torch.sum(neighbour_freq)

    return torch.squeeze(weights)

def cast_label_to_one_hot_and_prototype(fp_embd_w, fp_embd_s, y_noisy, weak_embed, strong_embed, noisy_labels, device='cpu', k=50, n_class=10, use_cosine_similarity=True, to_single_label=False):
    # Compute the label distribution for the noisy datasets
    max_prob_label_w, neighbour_label_distribution_w = label_distribution(
        query_embd=fp_embd_w,
        y_query=y_noisy,
        prior_embd=weak_embed,
        labels=noisy_labels,
        k=k,
        n_class=n_class,
        weighted=True,
        use_cosine_similarity=use_cosine_similarity
    )

    max_prob_label_s, neighbour_label_distribution_s = label_distribution(
        query_embd=fp_embd_s,
        y_query=y_noisy,
        prior_embd=strong_embed,
        labels=noisy_labels,
        k=k,
        n_class=n_class,
        weighted=True,
        use_cosine_similarity=use_cosine_similarity
    )

    loss_weights_w = get_loss_weights(fp_embd_w, y_noisy, weak_embed, noisy_labels, k=k, n_class=n_class)
    loss_weights_s = get_loss_weights(fp_embd_s, y_noisy, strong_embed, noisy_labels, k=k, n_class=n_class)

    kl_div = KL_label_distribution(neighbour_label_distribution_w.cpu(), neighbour_label_distribution_s.cpu())

    lower_set_batch, higher_set_batch = gmm_binary_split(kl_div)

    y_label_batch_w = sample_labels(
        neighbour_label_distribution_w,
        y_noisy,
        max_prob_label_w,
        lower_set_batch,
        higher_set_batch,
        to_single_label=to_single_label
    )

    y_label_batch_s = sample_labels(
        neighbour_label_distribution_s,
        y_noisy,
        max_prob_label_s,
        lower_set_batch,
        higher_set_batch,
        to_single_label=to_single_label
    )

    return y_label_batch_w, y_label_batch_s, loss_weights_w, loss_weights_s

