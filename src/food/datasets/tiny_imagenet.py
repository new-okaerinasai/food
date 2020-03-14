from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import cv2
import requests
import os
import zipfile
import numpy as np
from albumentations.pytorch import ToTensorV2 as ToTensor

class TinyImagenet(Dataset):
    def __init__(self, data_path: str, task="vanilla", mode="train", transform=ToTensor()):
        if task not in ["vanilla", "ood"]:
            raise RuntimeError(f"Unknown task {task}")
        self.transform = lambda x: transform(force_apply=True, image=x)["image"]

        data_path = os.path.abspath(data_path)

        if os.path.exists(data_path):
            print("Data path folder already exists. Continuing")
        else:
            print("Downloading files for TinyImagenet dataset")
            os.makedirs(data_path, exist_ok=True)
            request_res = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            zip_file = os.path.join(data_path, "tiny-imagenet-200.zip")
            with open(zip_file, "wb") as f:
                f.write(request_res.content)
            print("Done")
            print("Extracting")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(data_path)

        if mode == "train":
            tags_folders = os.path.join(data_path, "tiny-imagenet-200", mode)
            all_tags = sorted(os.listdir(tags_folders))
            num = 200 if task == "vanilla" else 100
            self.tag_2_class = {tag: cl for cl, tag in
                                enumerate(sorted(all_tags)[:num])}
            self.images_tags = [tag for tag in all_tags 
                                for im_name in os.listdir(os.path.join(tags_folders, tag, "images")) 
                                if tag in self.tag_2_class]
            self.images_fnames = [os.path.join(tags_folders, tag, "images", im_name) for tag in all_tags
                                  for im_name in os.listdir(os.path.join(tags_folders, tag, "images")) 
                                  if tag in self.tag_2_class]
        elif mode == "val":
            val_root = os.path.join(data_path, "tiny-imagenet-200", "val")
            annotations_fname = os.path.join(val_root, "val_annotations.txt")
            with open(annotations_fname, "r") as annotations_file:
                annotations = annotations_file.read().split("\n")
            self.images_fnames = [item.split("\t")[0] for item in annotations if len(item) > 1]
            self.images_fnames = [os.path.join(val_root, "images", image) for image in self.images_fnames]
            self.images_tags = [item.split("\t") for item in annotations]
            self.images_tags = [item[1] for item in self.images_tags if len(item) > 1]
            self.tag_2_class = {tag: (cl if task == "vanilla" else min(cl, 100))
                                for cl, tag in enumerate(sorted(np.unique(self.images_tags)))}                
        else:
            raise RuntimeError(f"Unknown mode {mode}")
        self.all_images = [cv2.imread(im) for im in self.images_fnames]

    def __getitem__(self, idx):
        #path = self.images_fnames[idx]
        image = self.all_images[idx] #cv2.imread(path)
        image = self.transform(image)
        return image, self.tag_2_class[self.images_tags[idx]]

    def __len__(self):
        return len(self.images_fnames)
