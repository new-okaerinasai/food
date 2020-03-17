import numpy as np
from torchvision.datasets import MNIST as MNIST_torchvision
from torchvision.datasets import FashionMNIST as FashionMNIST_torchvision
from albumentations.pytorch import ToTensorV2 as ToTensor


class MNIST(MNIST_torchvision):
    '''
    Attributes:
        class_to_id: dict, bijection from name of class to number of class 0 ... 9
        tag_to_class: dict, map from number of class 0 1 2 3 4 5 6 7 8 9
                            to new number of class 5 5 0 1 5 2 3 5 4 5
                            (0 1 2 3 4 5 6 7 8 9 if vanilla task)
    '''
    def __init__(self, datapath: str, task="vanilla", mode="train", transform=ToTensor()):
        if task not in ["vanilla", "ood"]:
            raise RuntimeError(f"Unknown task {task}")
        if mode not in ["train", "val"]:
            raise RuntimeError(f"Unknown mode {mode}")
        super().__init__(datapath, train=(mode == "train"), download=True)

        self.transform = lambda x: transform(force_apply=True,
                                             image=np.array(x)[..., None])["image"]
        self.tag_to_class = self.class_to_idx.copy()
        self.class_to_id = {cl : cl for cl in self.tag_to_class.values()}

        if task == "ood":
            not_ood_digits = [2, 3, 5, 6, 8]
            ood_digits = [0, 1, 4, 7, 9]
            self.class_to_id = {d : min(i, 5) for i, d in enumerate(not_ood_digits + ood_digits)}
            self.target_transform = lambda x: self.class_to_id[x]

            self.not_ood_digits = set(not_ood_digits)
            targets_numpy = self.targets.detach().numpy()
            not_ood_idx = np.where([x in self.not_ood_digits for x in targets_numpy])
            if mode == "train":
                self.data = self.data[not_ood_idx]
                self.targets = self.targets[not_ood_idx]

class FashionMNIST(FashionMNIST_torchvision):
    '''
    Attributes:
        class_to_id: dict, bijection from name of class to number of class 0 ... 9
        tag_to_class: dict, map from number of class 0 1 2 3 4 5 6 7 8 9
                            to new number of class 0 5 1 2 3 5 4 5 5 5
                            (0 1 2 3 4 5 6 7 8 9 if vanilla task)
    '''
    def __init__(self, datapath: str, task="vanilla", mode="train", transform=ToTensor()):
        if task not in ["vanilla", "ood"]:
            raise RuntimeError(f"Unknown task {task}")
        if mode not in ["train", "val"]:
            raise RuntimeError(f"Unknown mode {mode}")
        super().__init__(datapath, train=(mode == "train"), download=True)

        self.transform = lambda x: transform(force_apply=True,
                                             image=np.array(x)[..., None])["image"]
        self.tag_to_class = self.class_to_idx.copy()
        self.class_to_id = {cl : cl for cl in self.tag_to_class.values()}

        if task == "ood":
            not_ood_digits = [0, 2, 3, 4, 6]
            ood_digits = [1, 5, 7, 8, 9]
            self.class_to_id = {d : min(i, 5) for i, d in enumerate(not_ood_digits + ood_digits)}
            self.target_transform = lambda x: self.class_to_id[x]

            self.not_ood_digits = set(not_ood_digits)
            targets_numpy = self.targets.detach().numpy()
            not_ood_idx = np.where([x in self.not_ood_digits for x in targets_numpy])
            if mode == "train":
                self.data = self.data[not_ood_idx]
                self.targets = self.targets[not_ood_idx]
