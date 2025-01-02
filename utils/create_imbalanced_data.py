import random
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from utils.cifar_data_utils import *
from utils.ws_augmentation import TransformFixMatch_CIFAR10
import matplotlib.pyplot as plt


def create_imbalanced_dataset(dataset, reduce_fraction=0.2, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_classes = len(set(dataset.targets))
    selected_classes = random.sample(range(num_classes), num_classes // 2)
    
    data = []
    targets = []

    for cls in range(num_classes):
        cls_indices = [i for i, target in enumerate(dataset.targets) if target == cls]
        if cls in selected_classes:
            cls_indices = random.sample(cls_indices, max(1, int(len(cls_indices) * reduce_fraction)))
        data.extend([dataset.data[i] for i in cls_indices])
        targets.extend([dataset.targets[i] for i in cls_indices])

    new_dataset = Double_dataset(data=np.array(data), targets=np.array(targets), transform_fixmatch=dataset.transform_fixmatch)
    
    return new_dataset


def create_imbalanced_dataset_rho(dataset, rho=10, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_classes = len(set(dataset.targets))
    
    N = len(dataset.data) // num_classes
    class_sizes = []
    for k in range(1, num_classes + 1):
        class_size = int(N / (rho ** ((k - 1) / (num_classes - 1))))
        class_sizes.append(class_size)
    
    data = []
    targets = []
    
    for cls in range(num_classes):
        cls_indices = [i for i, target in enumerate(dataset.targets) if target == cls]
        num_samples = class_sizes[cls]
        selected_indices = random.sample(cls_indices, min(num_samples, len(cls_indices)))
        
        data.extend([dataset.data[i] for i in selected_indices])
        targets.extend([dataset.targets[i] for i in selected_indices])

    new_dataset = Double_dataset(data=np.array(data), targets=np.array(targets), transform_fixmatch=dataset.transform_fixmatch)
    
    return new_dataset


def main():

    train_dataset_cifar = datasets.CIFAR10(root='./data', train=True, download=False)

    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)
    transform_fixmatch = TransformFixMatch_CIFAR10(CIFAR_MEAN, CIFAR_STD, 2, 10)
    train_dataset = Double_dataset(
        data=train_dataset_cifar.data,
        targets=train_dataset_cifar.targets,
        transform_fixmatch=transform_fixmatch
    )

    imbalanced_dataset = create_imbalanced_dataset_rho(train_dataset, rho=10, seed=123)

if __name__ == "__main__":
    main()
