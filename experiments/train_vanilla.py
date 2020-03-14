import food
from food.datasets import TinyImagenet, CIFAR_100
from food.datasets.utils import DataPrefetcher

from logging_utils import log_dict_with_writer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, resnet50
from albumentations import Rotate, Compose, RandomBrightnessContrast, Normalize, HorizontalFlip
from albumentations.pytorch import ToTensorV2 as ToTensor

import numpy as np
import tqdm

import shutil
import argparse
import os
from typing import Tuple


def evaluate(model, dataloader, criterion, device) -> Tuple:
    """
    Evaluate model. This function prints validation loss and accuracy.
    :param model: model to evaluate
    :param dataloader: dataloader representing the validation data
    :param callable criterion: the loss function
    :return: None
    """
    all_predictions = []
    all_losses = []
    model = model.to(device).eval()
    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model.forward(images)
            batch_predictions = logits.argmax(dim=1)
            all_predictions.append((batch_predictions == labels).float())
            all_losses.append(criterion(logits, labels).item())
        accuracy = torch.cat(all_predictions).mean()
        loss = np.mean(all_losses)
        print("  Evaluation results: \n   Accuracy: {:.4f}\n   Loss: {:.4f}".format(accuracy, loss))
    return loss, accuracy

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data", type=str,
                        help="path where data is kept or will be downloaded")
    parser.add_argument("--batch_size", default=512, type=int,
                        help="um... batch size during training and testing, I guess?")
    parser.add_argument('--model', default="resnet18", type=str,
                        help="model to train. Must be resnet18 or resnet50")
    parser.add_argument('--dataset', default="tiny_imagenet", type=str,
                        help="dataset on which we train. Must be \'tiny_imagenet\' or 'cifar_100")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="learning rate. Must be float")
    parser.add_argument('--epochs', default=10, type=int,
                        help="number of training epochs")
    parser.add_argument('--logdir', default="./logs",
                        help="directory where logs for tensorboard will be stored. Training plots can be\
                         viewed with ```tensroboard --logdir=logdir```")
    parser.add_argument('--checkpoints_dir', default="./checkpoints",
                        help="directory where checkpoints will be stored")
    parser.add_argument('--checkpoint_each', default=20, type=int,
                        help="number of steps -- checkpointing interval")
    parser.add_argument('--log_each', default=10, type=int,
                        help="number of steps -- logging interval for tensorboard")
    parser.add_argument('--keep_logs', action="store_true",
                        help="set this to keep old logs in logdir if it exists")
    parser.add_argument('--resume', type=str, default=None,
                        help="path to a previous checkpointt to continue training")
    args = parser.parse_args()
    n_classes = {"tiny_imagenet": 200, "cifar_100": 100}

    if not args.keep_logs:
        try:
            shutil.rmtree(args.logdir)
        except FileNotFoundError:
            pass

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.model.lower() == "resnet18":
        train_transforms = Compose([
            RandomBrightnessContrast(p=0.5),
            Rotate(20),
            HorizontalFlip(0.5),
            Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ToTensor(),
        ], p=1)
        val_transforms = Compose([
            Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ToTensor(),
        ], p=1)
        model = resnet18(num_classes=n_classes[args.dataset.lower()])
    elif args.model.lower() == "resnet50":
        train_transforms = Compose([
            ToTensor()
        ])
        val_transforms = Compose([
            ToTensor()
        ])
        model = resnet50(num_classes=n_classes[args.dataset])
    else:
        raise NotImplementedError("Unknown model".format(args.model))
    model.to(device)

    if args.dataset.lower() == 'tiny_imagenet':
        train_dataset = food.datasets.TinyImagenet(args.data_path, mode="train", transform=train_transforms)
        val_dataset = food.datasets.TinyImagenet(args.data_path, mode="val", transform=val_transforms)
    elif args.dataset.lower() == "cifar_100":
        train_dataset = food.datasets.CIFAR_100(args.data_path, mode="train", transform=train_transforms)
        val_dataset = food.datasets.CIFAR_100(args.data_path, mode="train", transform=val_transforms)
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    #if torch.cuda.is_available():
        # optimize training with parallelism
        #train_dataloader = DataPrefetcher(train_dataloader)
        #val_dataloader = DataPrefetcher(val_dataloader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    train_writer = SummaryWriter(os.path.join(args.logdir, "train_logs"))
    val_writer = SummaryWriter(os.path.join(args.logdir, "val_logs"))
    global_step = 0
    for epoch in range(args.epochs):
        print("Training")
        for images, labels in tqdm.tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if global_step == 0:
                train_writer.add_graph(model, images)
            logits = model.forward(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            accuracy_t = torch.mean((predictions == labels).float()).item()
            if global_step % args.log_each == 0:
                train_writer.add_scalar("Loss_BCE", loss, global_step=global_step)
                train_writer.add_scalar("Accuracy", accuracy_t, global_step=global_step)
                train_writer.add_scalar("Learning_rate", scheduler.get_lr()[-1], global_step=global_step)
                #log_dict_with_writer(labels, logits, train_writer)
            loss.backward()
            optimizer.step()
            global_step += 1
        print("Validating...")
        val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
        val_writer.add_scalar("Loss_BCE", val_loss, global_step=global_step)
        val_writer.add_scalar("Accuracy", val_acc, global_step=global_step)
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        print("Saving checkpoint...")
        with open(os.path.join(args.checkpoints_dir, f"epoch_{epoch}_{global_step}.pt"), "wb") as f:
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, f)
        scheduler.step()

if __name__ == '__main__':
    train()
