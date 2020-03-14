from torchvision.datasets import CIFAR100
from albumentations.pytorch import ToTensorV2 as ToTensor
from torch.utils.data import Dataset


class CIFAR_100(Dataset):
    def __init__(self, data_path: str, mode="train", transform=ToTensor()):
        """
        This class wraps the torchvision CIFAR100. This is made to unify the class signature.
        :param data_path: Path to store the data
        :param mode: string, "train" or "val"
        """
        self.transform = lambda x: transform(force_apply=True, image=x)["image"]
        if mode == "train":
            self.cifar_100 = CIFAR100(data_path, train=True, download=True, transform=transform)
        elif mode == "val":
            self.cifar_100 = CIFAR100(data_path, train=False, download=True, transform=transform)
        else:
            raise ValueError("Unknown mode {}".format(mode))

    def __getitem__(self, item):
        return self.cifar_100[item]

    def __len__(self):
        return len(self.cifar_100)


__all__ = ['CIFAR100']

if __name__ == '__main__':
    cifar = CIFAR100("./data", transform=ToTensor(), download=True)
    print(cifar[0])
