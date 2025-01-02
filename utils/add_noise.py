import numpy as np
import random
import random
import torch
import numpy as np
import torchvision
from utils.cifar_data_utils import Custom_dataset, Double_dataset

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed: The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def add_symmetric_noise(targets, noise_ratio, num_classes, seed=None):
    if seed is not None:
        set_random_seed(seed)
    
    targets = np.array(targets)
    noisy_targets = targets.copy()
    n_samples = len(targets)
    n_noisy = int(noise_ratio * n_samples)
    noisy_indices = random.sample(range(n_samples), n_noisy)

    for idx in noisy_indices:
        original_label = targets[idx]
        possible_labels = list(range(num_classes))
        possible_labels.remove(original_label)
        new_label = random.choice(possible_labels)
        noisy_targets[idx] = new_label
    
    return noisy_targets

def add_asymmetric_noise(targets, noise_ratio, transition, seed=None):
    if seed is not None:
        set_random_seed(seed)

    targets = np.array(targets)
    noisy_targets = targets.copy()
    n_samples = len(targets)
    n_noisy = int(noise_ratio * n_samples)
    noisy_indices = random.sample(range(n_samples), n_noisy)

    for idx in noisy_indices:
        original_label = targets[idx]
        if original_label in transition:
            noisy_targets[idx] = transition[original_label]

    return noisy_targets

def add_noise(targets, noise_ratio, num_classes, seed=None, transition=None, symmetric_noise=True):
    if symmetric_noise:
        noisy_targets = add_symmetric_noise(targets, noise_ratio, num_classes, seed)
    else:
        noisy_targets = add_asymmetric_noise(targets, noise_ratio, transition, seed)
    
    return noisy_targets

def calculate_noise_rate(original_targets, noisy_targets):
    """
    Calculate the noise rate in the noisy targets.

    Parameters:
    - original_targets: The original labels.
    - noisy_targets: The noisy labels.

    Returns:
    - noise_rate: The proportion of noisy labels.
    """
    original_targets = np.array(original_targets)
    noisy_targets = np.array(noisy_targets)
    n_noisy = np.sum(original_targets != noisy_targets)
    noise_rate = n_noisy / len(original_targets)
    return noise_rate


def main():
    seed = 111
    set_random_seed(seed)

    train_dataset_cifar = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
    data = train_dataset_cifar.data
    targets = train_dataset_cifar.targets
    num_classes = 10
    noise_ratio = 0.4 

    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

    noisy_targets = add_noise(targets, noise_ratio, transition, num_classes, seed, symmetric_noise=True)

    transform_fixmatch = ... 
    train_dataset = Double_dataset(data=train_dataset_cifar.data[:], targets=train_dataset_cifar.targets[:], transform_fixmatch=transform_fixmatch)

    train_dataset.update_label(noisy_targets)
    print("Original labels: ", targets[:10])
    print("Noisy labels: ", noisy_targets[:10])

    actual_noise_rate = calculate_noise_rate(targets, noisy_targets)
    print(f"Actual noise rate: {actual_noise_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
