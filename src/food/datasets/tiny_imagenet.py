import os
import zipfile
import cv2
import requests
import numpy as np

from albumentations.pytorch import ToTensorV2 as ToTensor
# from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm

class TinyImagenet(Dataset):
    '''
    Attributes:
        class_to_id: dict, bijection from name of class to number of class 0 ... K (K / 2 if train)
        tag_to_class: dict, map from number of class 0 ... K to new number of class 0 ... K / 2 (K if vanilla task)
    '''
    def __init__(self, datapath: str, task="vanilla", mode="train", transform=ToTensor()):
        if task not in ["vanilla", "ood"]:
            raise RuntimeError(f"Unknown task {task}")
        self.transform = lambda x: transform(force_apply=True, image=x)["image"]

        datapath = os.path.abspath(datapath)

        if os.path.exists(os.path.join(datapath, "tiny-imagenet-200")):
            print("Data path folder already exists. Continuing")
        else:
            print("Downloading files for TinyImagenet dataset")
            os.makedirs(datapath, exist_ok=True)
            request_res = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            zip_file = os.path.join(datapath, "tiny-imagenet-200.zip")
            with open(zip_file, "wb") as f:
                f.write(request_res.content)
            print("Done")
            print("Extracting")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(datapath)
            os.remove(zip_file)
            print("Done")
        print("Reading data and target")

        self._images_classes = []
        self._images_fnames = []
        num = 200 if task == "vanilla" else 100
        self.n_classes = num
        if mode == "train":
            tags_folders = os.path.join(datapath, "tiny-imagenet-200", mode)

            all_tags = sorted(os.listdir(tags_folders))[:num]
            self.tag_to_class = {tag: cl for cl, tag in enumerate(all_tags)}

            for tag in all_tags:
                tag_path = os.path.join(tags_folders, tag, "images")
                tag_images_fnames = os.listdir(tag_path)
                for image_fname in tag_images_fnames:
                    self._images_classes.append(tag)
                    self._images_fnames.append(os.path.join(tag_path, image_fname))

        elif mode == "val":
            val_root = os.path.join(datapath, "tiny-imagenet-200", mode)
            annotations_fname = os.path.join(val_root, "val_annotations.txt")
            with open(annotations_fname, "r") as annotations_file:
                annotations = annotations_file.read().split("\n")

            all_tags = set()
            for item in annotations:
                items = item.split("\t")
                if len(item) > 1:
                    self._images_fnames.append(os.path.join(val_root, "images", items[0]))
                    self._images_classes.append(items[1])
                    all_tags.add(items[1])

            self.tag_to_class = {tag: cl for cl, tag in enumerate(sorted(all_tags))}
        else:
            raise RuntimeError(f"Unknown mode {mode}")

        self.class_to_id = {cl: (idx if task == "vanilla" else min(idx, 100))
                                for idx, cl in enumerate(range(len(self.tag_to_class)))}
        self.data = [cv2.imread(im)[..., ::-1] for im in tqdm(self._images_fnames)]
        self.targets = [self.class_to_id[self.tag_to_class[cl]] for cl in self._images_classes]

    def __getitem__(self, item):
        image, target = self.data[item], self.targets[item]
        image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.targets)

