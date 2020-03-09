import food
from food.datasets import TinyImagenet

import torch
from torch import nn
import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, resnet50
import numpy as np
import shutil

from typing import Tuple
import argparse
import os

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
    model = model.to(device)
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
        print("  Evaluation results: \n    Accuracy: {:.4f}\n   Loss: {:.4f}".format(accuracy, loss))
    return loss, accuracy

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument('--model', default="resnet50", type=str,
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
    args = parser.parse_args()
    try:
        shutil.rmtree(args.logdir)
    except FileNotFoundError:
        pass

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.dataset.lower() == 'tiny_imagenet':
        train_dataset = food.TinyImagenet(args.data_path, mode="train")
        val_dataset = food.TinyImagenet(args.data_path, mode="val")
    elif args.dataset.lower() == "cifar_100":
        train_dataset = food.CIFAR100(args.data_path, train=True, download=True, transform=ToTensor())
        val_dataset = food.CIFAR100(args.data_path, train=False, download=True, transform=ToTensor())
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.model.lower() == "resnet18":
        model = resnet18()
    elif args.model.lower() == "resnet50":
        model = resnet50()
    else:
        raise NotImplementedError("Unknown model".format(args.model))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.logdir)
    global_step = 0
    exp_mean_acc = None
    for epoch in range(args.epochs):
        print("Training")
        for images, labels in tqdm.tqdm(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if global_step == 0:
                writer.add_graph(model, images)
            logits = model.forward(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            accuracy_t = torch.mean((predictions == labels).float()).item()
            if exp_mean_acc:
                exp_mean_acc = accuracy_t * 0.9 + exp_mean_acc * 0.1
            else:
                exp_mean_acc = accuracy_t
            if global_step % args.log_each == 0:
                writer.add_scalar("Train/Loss_BCE", loss, global_step=global_step)
                writer.add_scalar("Train/Accuracy", exp_mean_acc, global_step=global_step)
                writer.add_scalar("Learning_rate", scheduler.get_lr()[-1], global_step=global_step)
            if global_step % args.checkpoint_each == 0:
                os.makedirs(args.checkpoints_dir, exist_ok=True)
                with open(os.path.join(args.checkpoints_dir, f"epoch_{epoch}_{global_step}.pt"), "wb") as f:
                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()}, f)
            loss.backward()
            optimizer.step()
            global_step += 1
        print("Validating...")
        val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
        writer.add_scalar("Val/Loss_BCE", val_loss, global_step=global_step)
        writer.add_scalar("Val/Accuracy", val_acc, global_step=global_step)
        scheduler.step()

if __name__ == '__main__':
    train()
