from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import cv2
import requests
import os
import zipfile
import numpy as np

class TinyImagenet(Dataset):
    def __init__(self, data_path: str, mode="train"):
        self.transform = ToTensor()
        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path):
            print("Data path folder already exists. Continuing")
        else:
            print("Downloading files for TinyImagenet dataset")
            os.makedirs(data_path, exist_ok=True)
            request_res = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            zip_file = os.path.join(data_path, "tiny-imagenet-200.zip")
            with open(zip_file, 'wb') as f:
                f.write(request_res.content)
            print("Done")
            print("Extracting")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_path)
        if mode == "train":
            classes_folders = os.path.join(data_path, "tiny-imagenet-200", mode)
            all_classes = sorted(os.listdir(classes_folders))
            self.images_classes = [cl for cl in all_classes for im_name in
                                   os.listdir(os.path.join(classes_folders, cl, "images"))]
            self.tag_2_class = {tag: cl for cl, tag in
                                enumerate(sorted(all_classes))}
            self.images_fnames = [os.path.join(classes_folders, cl, "images", im_name) for cl in
                                  all_classes for im_name in
                                  os.listdir(os.path.join(classes_folders, cl, "images"))]
        elif mode == "val":
            val_root = os.path.join(data_path, "tiny-imagenet-200", "val")
            annotations_fname = os.path.join(val_root, "val_annotations.txt")
            annotations = open(annotations_fname, "r").read().split("\n")
            self.images_fnames = [item.split("\t")[0] for item in annotations if len(item) > 1]
            self.images_fnames = [os.path.join(val_root, "images", image) for image in self.images_fnames]
            self.images_classes = [item.split("\t") for item in annotations]
            self.images_classes = [item[1] for item in self.images_classes if len(item) > 1]
            self.tag_2_class = {tag: cl for cl, tag in enumerate(sorted(np.unique(self.images_classes)))}
        else:
            raise RuntimeError(f"Unknown mode {mode}")

    def __getitem__(self, item):
        path = self.images_fnames[item]
        image = cv2.imread(path)
        image = self.transform(image)
        return image, self.tag_2_class[self.images_classes[item]]

    def __len__(self):
        return len(self.images_fnames)


if __name__ == "__main__":
    ds = TinyImagenet("./data", mode="train")
