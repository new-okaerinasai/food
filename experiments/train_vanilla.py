import food
from food.datasets import TinyImagenet, CIFAR_100
from utils import Config

from logging_utils import log_dict_with_writer, log_hist_as_picture

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, resnet50, mobilenet_v2
import torchvision
from albumentations import (Rotate, Compose, RandomBrightnessContrast,
                            Normalize, HorizontalFlip, VerticalFlip,
                            RandomResizedCrop, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2 as ToTensor

import numpy as np
import tqdm

import shutil
import argparse
import os
from typing import Tuple


def evaluate(model, dataloader, criterion, device, writer, T=None, eps=None) -> Tuple:
    """
    Evaluate model. This function prints validation loss and accuracy.
    :param model: model to evaluate
    :param dataloader: dataloader representing the validation data
    :param callable criterion: the loss function
    :return: None
    """
    model = model.to(device).eval()
    all_predictions = []
    all_losses = []
    all_logits = []
    all_labels = []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).long()
        if T is not None:
            # ODIN case: this is simple
            # Just make a slight adversarial attack on the images
            # source: https://arxiv.org/pdf/1706.02690.pdf
            images.requires_grad = True
            logits = model.forward(images)
            logits = logits / T
            pred_labels = logits.argmax(-1)
            pred_loss = criterion(logits, pred_labels)
            pred_loss.backward()
            corrupted_images = images - eps * torch.sign(-(images.grad))
            logits = model.forward(corrupted_images)
        else:
            logits = model.forward(images)

        all_logits.append(logits)
        all_labels.append(labels)
        # just not to write additional code for ood and vanilla task
        # valid mask corresponds only for known labels
        valid_logits_mask = (labels < logits.shape[1])
        valid_logits = logits[valid_logits_mask]
        valid_labels = labels[valid_logits_mask]
        valid_predictions = valid_logits.argmax(1)
        all_predictions.append((valid_predictions == valid_labels).float())
        all_losses.append(criterion(valid_logits, valid_labels).item())
    accuracy = torch.cat(all_predictions).mean()
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    print(all_labels.shape, all_logits.shape)
    log_hist_as_picture(all_labels, all_logits, ood_label=logits.shape[1])
    loss = np.mean(all_losses)
    print("  Evaluation results: \n  Accuracy: {:.4f}\n  Loss: {:.4f}".format(
        accuracy, loss))
    return loss, accuracy


def train(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./vanilla.json", type=str,
                        help="path to the config file")
    args = parser.parse_args()
    args = Config(args.config)
    get_dataset_with_arg = {"tiny_imagenet": food.datasets.TinyImagenet,
                            "cifar_100": food.datasets.CIFAR_100,
                            "cifar_10": food.datasets.CIFAR_10}
    if not args.keep_logs:
        try:
            shutil.rmtree(args.logdir)
        except FileNotFoundError:
            pass

    batch_size = args.batch_size
    batch_size = kwargs.get('batch_size', batch_size)

    model = args.model.lower()
    model = kwargs.get('model', model).lower()

    dataset = args.dataset.lower()
    dataset = kwargs.get('dataset', dataset).lower()

    epochs = args.epochs
    epochs = kwargs.get('epochs', epochs)

    test_b = kwargs.get('test', False)

    ds_class = get_dataset_with_arg[dataset.lower()]

    train_transforms = Compose([
        HorizontalFlip(p=0.5),
        RandomResizedCrop(32, 32, p=0.5),
        Rotate(15),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ], p=1)

    val_transforms = Compose([
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ], p=1)

    train_dataset = ds_class(args.data_path, mode="train", task=args.task.lower(),
                             transform=train_transforms)
    val_dataset = ds_class(args.data_path, mode="val", task=args.task.lower(),
                                                transform=val_transforms)
    model = getattr(torchvision.models, args.model)(pretrained=True)
    print(model)
    # The last fully-connected layers in torchvision resnet are called fc
    model.fc = nn.Linear(model.fc.in_features, train_dataset.n_classes)
    print(model.__class__.__name__)
    print("Total number of model's parameters: ",
          np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = 0.2, threshold = 0.002, patience=10, threshold_mode='abs')

    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        resume_epoch = state_dict["epoch"]

    os.makedirs(os.path.join(args.logdir, "train_logs"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "val_logs"), exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.logdir, "train_logs"))
    val_writer = SummaryWriter(os.path.join(args.logdir, "val_logs"))
    global_step = 0
    for epoch in range(resume_epoch, epochs):
        print(f"Training, epoch {epoch + 1}")
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            logits = model.forward(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            accuracy_t = torch.mean((predictions == labels).float()).item()
            if global_step % args.log_each == 0:
                # TODO: train logging for tensorboard
                # train_writer.add_scalar("Loss_BCE", loss, global_step=global_step)
                # train_writer.add_scalar("Accuracy", accuracy_t, global_step=global_step)
                # train_writer.add_scalar("Learning_rate", scheduler.get_lr()[-1], global_step=global_step)
                #log_dict_with_writer(labels, logits, train_writer, global_step=global_step)
                pass
            loss.backward()
            optimizer.step()
            global_step += 1
        print("Validating...")
        val_loss, val_acc = evaluate(
            model, val_dataloader, criterion, device, val_writer, T=args.temperature, eps=args.eps)
        val_writer.add_scalar("Loss_BCE", val_loss, global_step=global_step)
        val_writer.add_scalar("Accuracy", val_acc, global_step=global_step)

        if epoch % args.checkpoint_each == 0:
            print("Saving checkpoint...")
            with open(os.path.join(args.checkpoints_dir, f"epoch_{epoch}_{global_step}.pt"), "wb") as f:
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "scheduler_state_dict": scheduler.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()}, f)
        scheduler.step(val_acc)

    if test_b:
        return logits, loss, predictions, val_loss, val_acc


if __name__ == '__main__':
    train()
