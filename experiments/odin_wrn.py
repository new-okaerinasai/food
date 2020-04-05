import food
from food.datasets import TinyImagenet, CIFAR_100
from wrn import wrn
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
import torch.nn.functional as F

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import tqdm

import shutil
import argparse
import json
import os
from typing import Tuple


def evaluate(model, val_dataloader, ood_dataloader, criterion, device, writer, T=None, eps=None) -> Tuple:
    """
    Evaluate model. This function prints validation loss and accuracy.
    :param model: model to evaluate
    :param dataloader: dataloader representing the validation data
    :param callable criterion: the loss function
    :return: None
    """
    # Validation
    model = model.to(device).eval()
    all_predictions = []
    all_losses = []
    all_logits = []
    all_labels = []
    print("Temerature = ", T)
    print("Epsilon = ", eps)
    
    # OOD
    all_ood_probs = []
    for i, (images, labels) in enumerate(tqdm.tqdm(ood_dataloader)):
        if i > 625:
            break
        images, labels = images.to(device), labels.to(device).long()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
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
            images.grad.zero_()
        else:
            logits = model.forward(images)

        all_ood_probs.append(F.softmax(logits, dim=1).cpu().detach().numpy())

    all_ood_probs = np.concatenate(all_ood_probs).max(1)
    for images, labels in tqdm.tqdm(val_dataloader):
        images, labels = images.to(device), labels.to(device).long()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
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

        labels = labels.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()
        all_logits.append(logits)
        all_labels.append(labels)
        # just not to write additional code for ood and vanilla task
        # valid mask corresponds only for known labels
        valid_logits_mask = (labels < logits.shape[1])
        valid_logits = logits[valid_logits_mask]
        valid_labels = labels[valid_logits_mask]
        valid_predictions = valid_logits.argmax(1)
        all_predictions.append((valid_predictions == valid_labels))
        all_losses.append(criterion(torch.from_numpy(valid_logits), torch.from_numpy(valid_labels)).item())
        torch.cuda.empty_cache()

    accuracy = np.concatenate(all_predictions).mean()
    all_probs = softmax(np.concatenate(all_logits), axis=1).max(1)
    all_labels = np.concatenate(all_labels)
    loss = np.mean(all_losses)
    print("  Evaluation results: \n  Accuracy: {:.4f}\n  Loss: {:.4f}".format(
        accuracy, loss))
    
    plt.figure(figsize=(10, 8))
    plt.title("Known classes vs OOD max logit distribution")
    plt.hist(all_probs, range=(0,1), bins=20, color="blue", alpha=0.5, label="max_softmax_probability_true", density=True)
    plt.hist(all_ood_probs, range=(0,1), bins=20, color="red", alpha=0.5, label="max_softmax_probability_ood", density=True)
    plt.legend()
    plt.savefig("hist.png")
    #log_hist_as_picture(all_labels, all_logits, ood_label=logits.shape[1])

    return loss, accuracy


def train(**kwargs):
    if len(kwargs) == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="./vanilla.json", type=str,
                            help="path to the config file")
        args = parser.parse_args()
        args = Config(args.config)
    else:
        args = Config(kwargs.get("config", "./vanilla.json"))
    print(json.dumps(args.__dict__, indent=4))
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
        #RandomResizedCrop(32, 32, p=0.5),
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
    ood_dataset = TinyImagenet(args.data_path, mode="train", task="vanilla", transform=val_transforms)

    model = wrn(depth=28, num_classes=train_dataset.n_classes, widen_factor=10, dropRate=0.3)
    #getattr(torchvision.models, args.model)(pretrained=True)
    print(model)
    # The last fully-connected layers in torchvision resnet are called fc
    # model.fc = nn.Linear(model.fc.in_features, train_dataset.n_classes)
    print(model.__class__.__name__)
    print("Total number of model's parameters: ",
          np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    ood_dataloader = DataLoader(
        ood_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = 0.2, threshold = 0.002, patience=10, threshold_mode='abs')

    resume_epoch = 0
    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location=device)
        state_dict = state_dict["state_dict"]
        old_state_dict = model.state_dict()
        for key in state_dict:
            old_state_dict[key[7:]] = state_dict[key]
        model.load_state_dict(old_state_dict)
        # model.load_state_dict(state_dict["model_state_dict"])
        # optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        # scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        # resume_epoch = state_dict["epoch"]

    os.makedirs(os.path.join(args.logdir, "train_logs"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "val_logs"), exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.logdir, "train_logs"))
    val_writer = SummaryWriter(os.path.join(args.logdir, "val_logs"))
    global_step = 0
    for epoch in range(resume_epoch, epochs):
        print(f"Training, epoch {epoch + 1}")
        model.train()
        evaluate(
            model, val_dataloader, ood_dataloader, criterion, device, val_writer, T=args.temperature, eps=args.eps)
        exit(0)
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
            model, val_dataloader, ood_dataloader, criterion, device, val_writer, T=args.temperature, eps=args.eps)
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
