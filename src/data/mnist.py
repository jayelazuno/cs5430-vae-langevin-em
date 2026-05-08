import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(root, batch_size, num_workers=2, val_fraction=0.1, seed=123):
    transform = transforms.ToTensor()

    full_train = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train,
        [n_train, n_val],
        generator=generator,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
