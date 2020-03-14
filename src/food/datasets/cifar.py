import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100


class CIFAR_10(CIFAR10):
    def __init__(self, datapath: str, task="vanilla", mode="train"):
        super().__init__(datapath, train=(mode == "train"), download=True)
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
            self.class_to_idx = {cl : min(idx, 5) for cl, idx in self.class_to_idx.items()}
        elif task == "vanilla":
            pass
        else:
            raise RuntimeError(f"Unknown task {task}")


class CIFAR_100(CIFAR100):
    def __init__(self, datapath: str, task="vanilla", mode="train"):
        super().__init__(datapath, train=(mode == "train"), download=True)
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
            self.class_to_idx = {cl : min(idx, 50) for cl, idx in self.class_to_idx.items()}
        elif task == "vanilla":
            pass
        else:
            raise RuntimeError(f"Unknown task {task}")
