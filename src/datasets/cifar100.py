from torchvision.datasets import CIFAR100
from torchvision.transforms import  ToTensor

if __name__ == '__main__':
    cifar = CIFAR100("./data", transform=ToTensor(), download=True)
    print(cifar[0])