import food
from food.datasets import TinyImagenet, MNIST, FashionMNIST, CIFAR_10, CIFAR_100
import torch

def test_mnist():
    ds = MNIST("./data", mode="train")
    for img, label in ds:
        assert isinstance(img, torch.Tensor)
        assert img.shape == (1, 28, 28)
        assert isinstance(label, int)
        assert label >= 0 and label <= 9

    ds_ood = MNIST("./data", mode="val", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label <= 5

    ds_ood = MNIST("./data", mode="train", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label < 5

def test_fashion_mnist():
    ds = FashionMNIST("./data", mode="train")
    assert len(ds[0]) == 2
    for img, label in ds:
        assert isinstance(img, torch.Tensor)
        assert img.shape == (1, 28, 28)
        assert isinstance(label, int)
        assert label >= 0 and label <= 9

    ds_ood = FashionMNIST("./data", mode="val", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label <= 5

    ds_ood = FashionMNIST("./data", mode="train", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label < 5

def test_cifar_10():
    ds = CIFAR_10("./data", mode="train")
    assert len(ds[0]) == 2
    for img, label in ds:
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert label >= 0 and label <= 9

    ds_ood = CIFAR_10("./data", mode="val", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label <= 5

    ds_ood = CIFAR_10("./data", mode="train", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label < 5

def test_cifar_100():
    ds = CIFAR_100("./data", mode="train")
    assert len(ds[0]) == 2
    for img, label in ds:
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert label >= 0 and label <= 99

    ds_ood = CIFAR_100("./data", mode="val", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label <= 50

    ds_ood = FashionMNIST("./data", mode="train", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label < 50

def test_tiny_imagenet():
    ds = TinyImagenet("./data", mode="train")
    assert len(ds[0]) == 2
    for img, label in ds:
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert label >= 0 and label <= 199

    ds_ood = TinyImagenet("./data", mode="val", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label <= 100

    ds_ood = FashionMNIST("./data", mode="train", task="ood")
    for img, label in ds_ood:
        assert label >= 0 and label < 100

