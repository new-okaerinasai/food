import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from albumentations.pytorch import ToTensorV2 as ToTensor


class CIFAR_10(CIFAR10):
    '''
    Attributes:
        class_to_id: dict, bijection from name of class to number of class 0 ... K
        tag_to_class: dict, map from number of class 0 ... K to new number of class 0 ... K / 2 (K if vanilla task)
    '''
    def __init__(self, datapath: str, task="vanilla", mode="train", transform=ToTensor()):
        if task not in ["vanilla", "ood"]:
            raise RuntimeError(f"Unknown task {task}")
        super().__init__(datapath, train=(mode == "train"), download=True)
        self.transform = lambda x: transform(force_apply=True,
                                             image=np.array(x))["image"]
        self.tag_to_class = self.class_to_idx.copy()
        self.class_to_id = {cl : cl for cl in self.tag_to_class.values()}

        if task == "ood":
            self.targets = np.array(self.targets)
            not_ood = (self.targets < 5)
            if mode == "train":
                self.data = self.data[not_ood]
                self.targets = self.targets[not_ood]
            elif mode == "val":
                self.targets[self.targets > 5] = 5
            else:
                raise RuntimeError(f"Unknown mode {mode}")
            self.class_to_id = {idx : min(idx, 5) for cl, idx in self.class_to_idx.items()}


class CIFAR_100(CIFAR100):
    '''
    Attributes:
        class_to_id: dict, bijection from name of class to number of class 0 ... K
        tag_to_class: dict, map from number of class 0 ... K to new number of class 0 ... K / 2 (K if vanilla task)
    '''
    def __init__(self, datapath: str, task="vanilla", mode="train", transform=ToTensor()):
        if task not in ["vanilla", "ood"]:
            raise RuntimeError(f"Unknown task {task}")
        super().__init__(datapath, train=(mode == "train"), download=True)
        self.transform = lambda x: transform(force_apply=True,
                                             image=np.array(x))["image"]
        self.tag_to_class = self.class_to_idx.copy()
        self.class_to_id = {cl : cl for cl in self.tag_to_class.values()}

        if task == "ood":
            self.targets = np.array(self.targets)
            not_ood = (self.targets < 50)
            if mode == "train":
                self.data = self.data[not_ood]
                self.targets = self.targets[not_ood]
            elif mode == "val":
                self.targets[self.targets > 50] = 50
            else:
                raise RuntimeError(f"Unknown mode {mode}")
            self.class_to_id = {idx : min(idx, 50) for cl, idx in self.class_to_idx.items()}
