import food
from utils import Config, encode_onehot, ConfidenceNet, GrayModel

from logging_utils import log_dict_with_writer, log_hist_as_picture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models import resnet18, resnet50, mobilenet_v2

from albumentations import (Rotate, Compose, RandomBrightnessContrast,
                            Normalize, HorizontalFlip, VerticalFlip,
                            RandomResizedCrop, ShiftScaleRotate, Resize, Lambda)
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image

import numpy as np
import tqdm

import shutil
import argparse
import os
from typing import Tuple


def evaluate(model, dataloader, criterion, device, logdir) -> Tuple:
    """
    Evaluate model. This function prints validation loss and accuracy.
    :param model: model to evaluate
    :param dataloader: dataloader representing the validation data
    :param callable criterion: the loss function
    :return: None
    """
    all_predictions = []
    all_logits = []
    all_labels = []
    all_confs =[]

    model = model.to(device).eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            logits, conf = model.forward(images)

            # just not to write additional code for ood and vanilla task
            # valid mask corresponds only for known labels
            valid_logits_mask = (labels < logits.shape[1])
            valid_logits = logits[valid_logits_mask]
            valid_labels = labels[valid_logits_mask]

            pred = F.softmax(valid_logits, dim=-1)
            conf = torch.sigmoid(conf).view(-1)

            pred = torch.argmax(pred, 1)
            all_predictions.append((pred == valid_labels).float())

            all_logits.append(logits)
            all_labels.append(labels)
            all_confs.append(conf)

        accuracy = torch.cat(all_predictions).mean()
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        all_confs = torch.cat(all_confs)
        print(all_logits, all_labels, all_confs)
        print(all_logits.shape, all_labels.shape, all_confs.shape)
        conf_min, conf_max, conf_avg = all_confs.min(), all_confs.max(), all_confs.mean()
        log_hist_as_picture(all_labels, all_logits, os.path.join(logdir, "hist_logits.png"), ood_label=logits.shape[1])
        log_hist_as_picture(all_labels, all_confs, os.path.join(logdir, "hist_confs.png"), ood_label=logits.shape[1])
        
        print("  Evaluation results: \n  Accuracy: {:.4f}\n "
              "  Confidence:\n\tmin {:.4f}\n\tmax {:.4f}\n\tmean {:.4f}".format(accuracy, conf_min, conf_max, conf_avg))
    return conf_min, conf_max, conf_avg, accuracy


def train(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./vanilla.json", type=str,
                        help="path to the config file")
    args = parser.parse_args()
    args = Config(args.config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    get_dataset_with_arg = {"tiny_imagenet": food.datasets.TinyImagenet,
                            "cifar_100": food.datasets.CIFAR_100,
                            "cifar_10": food.datasets.CIFAR_10,
                            "mnist": food.datasets.MNIST,
                            "fmnist": food.datasets.FashionMNIST}
    if not args.keep_logs:
        try:
            shutil.rmtree(args.logdir)
        except FileNotFoundError:
            pass

    batch_size = kwargs.get('batch_size', args.batch_size)
    base_model = kwargs.get('model', args.model).lower()
    dataset_train = kwargs.get('dataset_train', args.dataset_train).lower()
    dataset_test = kwargs.get('dataset_test', args.dataset_test).lower()
    epochs = kwargs.get('epochs', args.epochs)
    # test_b = kwargs.get('test', False)

    ds_class_train = get_dataset_with_arg[dataset_train]
    ds_class_test = get_dataset_with_arg[dataset_test]

    p_flip = 0.5 if dataset_train != "mnist" else 0
    norm_train = Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    norm_test = norm_train
    if 'mnist' in dataset_train:
        norm_train = Normalize((0.1307,), (0.3081,))
    if 'mnist' in dataset_test:
        norm_test = Normalize((0.1307,), (0.3081,))
    train_transforms = Compose([HorizontalFlip(p=p_flip),
                                RandomResizedCrop(32, 32, p=0.5),
                                norm_train,
                                ToTensor()])

    val_transforms = Compose([norm_test,
                              ToTensor()])

    train_dataset = ds_class_train(args.data_path, mode="train", task=args.task.lower(),
                                                transform=train_transforms)
    val_dataset = ds_class_test(args.data_path, mode="val", task=args.task.lower(),
                                                transform=val_transforms)

    model = getattr(torchvision.models, base_model)(num_classes=5)
    if 'mnist' in dataset_train:
        model = GrayModel(model)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # ==================================================================================

    model = ConfidenceNet(model, 5).to(device)
    
    print(model)
    print("Total number of model's parameters: ",
          np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # ==================================================================================
 
    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.logdir, "train_logs"))
    val_writer = SummaryWriter(os.path.join(args.logdir, "val_logs"))
    global_step = 0

    # ==================================================================================

    lmbda = 0.1  # initial lambda
    for epoch in range(epochs):

        print(f"Training, epoch {epoch + 1}")
        model.train()
        # for images, labels in list(train_dataloader)[:3]:
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            labels_oh = encode_onehot(labels, 5)
            optimizer.zero_grad()
            
            logits, confidence = model.forward(images)
            pred = F.softmax(logits, dim=-1)
            confidence = torch.sigmoid(confidence)

            c = torch.bernoulli(torch.rand(confidence.size())).to(device)  # uniformly random number
            conf = confidence * c + (1 - c)
            pred_new = pred * conf.expand_as(pred) + labels_oh * (1 - conf.expand_as(labels_oh))
            pred_new = torch.log(pred_new)

            loss_xe = criterion(pred_new, labels)
            loss_confidence = torch.mean(-torch.log(confidence))
            loss = loss_xe + lmbda * loss_confidence
            if args.budget > loss_confidence:
                lmbda /= 1.01
            else:
                lmbda /= 0.99
            
            predictions = pred.argmax(dim=1)
            accuracy_t = torch.mean((predictions == labels).float()).item()
            if global_step % args.log_each == 0:
                # TODO
                train_writer.add_scalar("Loss_CE", loss_xe, global_step=global_step)
                train_writer.add_scalar("Loss_Conf", loss_confidence, global_step=global_step)
                train_writer.add_scalar("Accuracy", accuracy_t, global_step=global_step)
                train_writer.add_scalar("Lambda", lmbda, global_step=global_step)
                train_writer.add_scalar("Learning_rate", scheduler.get_lr()[-1], global_step=global_step)
                # log_dict_with_writer(labels, logits, train_writer, global_step=global_step)
            loss.backward()
            optimizer.step()
            global_step += 1
        print("Validating...")
        conf_min, conf_max, conf_avg, val_acc = evaluate(model, val_dataloader, criterion, device, args.logdir)
        val_writer.add_image("Logits", torchvision.transforms.ToTensor()(
            Image.open(os.path.join(args.logdir, "hist_logits.png")).convert("RGB")), global_step=global_step)
        val_writer.add_image("Confidences", torchvision.transforms.ToTensor()(
            Image.open(os.path.join(args.logdir, "hist_confs.png")).convert("RGB")), global_step=global_step)
        val_writer.add_scalar("Conf", conf_avg, global_step=global_step)
        val_writer.add_scalar("Conf", conf_min, global_step=global_step)
        val_writer.add_scalar("Conf", conf_max, global_step=global_step)
        val_writer.add_scalar("Conf", conf_avg, global_step=global_step)
        val_writer.add_scalar("Accuracy", val_acc, global_step=global_step)

        if (epoch + 1) % args.checkpoint_each == 0:  # prevent not saving 999 epoch 
            print("Saving checkpoint...")
            with open(os.path.join(args.checkpoints_dir, f"epoch_{epoch}_{global_step}.pt"), "wb") as f:
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()}, f)
        scheduler.step()

    # if test_b:
    #     return logits, loss, predictions, val_loss, val_acc

if __name__ == '__main__':
    train()
