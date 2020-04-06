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
from albumentations import (Rotate, Resize, Compose, RandomBrightnessContrast,
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

from sklearn.metrics import (precision_recall_curve, roc_auc_score, f1_score, auc)


def evaluate(model, val_dataloader, ood_dataloader, criterion, device, writer,
T=None, eps=None) -> Tuple:
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

    is_ood = []
    # OOD
    all_ood_probs = []
    print("Temperature =", T)
    for i, (images, labels) in enumerate(tqdm.tqdm(ood_dataloader)):
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
        all_ood_probs.append(
            F.softmax(logits, dim=1).cpu().detach().numpy())
        is_ood.append(np.zeros(len(images)))

    # Known classes -- for validation
    for i, (images, labels) in enumerate(tqdm.tqdm(val_dataloader)):
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
        all_losses.append(criterion(torch.from_numpy(
            valid_logits), torch.from_numpy(valid_labels)).item())
        torch.cuda.empty_cache()
        is_ood.append(np.ones(len(images)))
    accuracy = np.concatenate(all_predictions).mean()
    all_probs = softmax(np.concatenate(all_logits), axis=1).max(1)
    all_ood_probs = np.concatenate(all_ood_probs).max(1)
    y_pred_ood = np.concatenate([all_ood_probs, all_probs])
    is_ood = np.concatenate(is_ood)
    all_labels = np.concatenate(all_labels)
    loss = np.mean(all_losses)
    a, b, _ = precision_recall_curve(is_ood, y_pred_ood)
    print("""  Evaluation results: 
                 Accuracy: {:.4f}
                 Loss: {:.4f}
                 AUC-ROC: {:.4f}
                 PR-AUC: {:.4f}
    """.format(
        accuracy,
        loss,
        roc_auc_score(is_ood, y_pred_ood),
        auc(np.sort(a), np.sort(b)[::-1])))

    plt.figure(figsize=(10, 8))
    plt.title("Known classes vs OOD max logit distribution")
    plt.hist(all_probs, range=(0, 1), bins=20, color="blue", alpha=0.5,
             label="max_softmax_probability_true", density=True)
    plt.hist(all_ood_probs, range=(0, 1), bins=20, color="red",
             alpha=0.5, label="max_softmax_probability_ood", density=True)
    plt.legend()
    plt.savefig("hist_oecc.png")
    # log_hist_as_picture(all_labels, all_logits, ood_label=logits.shape[1])
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

    if args.dataset == 'cifar_10':
        A_tr = 0.9500  # Training accuracy of CIFAR-10 baseline model
    elif args.dataset == 'cifar_100':
        A_tr = 0.8108

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
        # RandomResizedCrop(32, 32, p=0.5),
        Rotate(15),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ], p=1)

    val_transforms = Compose([
        Resize(32, 32),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ], p=1)

    train_dataset = ds_class(args.data_path, mode="train", task=args.task.lower(),
                             transform=train_transforms)
    val_dataset = ds_class(args.data_path, mode="val", task=args.task.lower(),
                                                transform=val_transforms)
    ood_dataset_train = TinyImagenet(
        args.data_path, mode="train", task="vanilla", transform=val_transforms)
    ood_dataset_val = TinyImagenet(
        args.data_path, mode="val", task="vanilla", transform=val_transforms)

    model = wrn(depth=28, num_classes=train_dataset.n_classes,
                widen_factor=10, dropRate=0.3)
    print(model)
    print(model.__class__.__name__)
    print("Total number of model's parameters: ",
          np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    ood_dataloader_train = DataLoader(
        ood_dataset_train, batch_size=batch_size, shuffle=False, drop_last=False)
    ood_dataloader_val = DataLoader(
        ood_dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=0.9,
        weight_decay=0.0005, nesterov=True)

    # Learning Rate
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_dataloader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.lr))

    resume_epoch = 0
    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location=device)
        state_dict = state_dict["state_dict"]
        old_state_dict = model.state_dict()
        for key in state_dict:
            old_state_dict[key[7:]] = state_dict[key]
        model.load_state_dict(old_state_dict)

    os.makedirs(os.path.join(args.logdir, "train_logs"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "val_logs"), exist_ok=True)
    # train_writer = SummaryWriter(os.path.join(args.logdir, "train_logs"))
    val_writer = SummaryWriter(os.path.join(args.logdir, "val_logs"))
    global_step = 0
    for epoch in range(resume_epoch, epochs):
        print(f"Training, epoch {epoch + 1}")
        model.train()
        val_loss, val_acc = evaluate(
            model, val_dataloader, ood_dataloader_val, criterion, device, val_writer,
            T=args.temperature, eps=args.eps)
        exit(0)
        for (images, labels), (ood_images, ood_labels) in tqdm.tqdm(zip(train_dataloader, ood_dataloader_train)):
            images, labels = images.to(device), labels.to(device).long()
            ood_images = ood_images.to(device)
            optimizer.zero_grad()
            # predict in-distribution probabilities
            logits_in = model.forward(images)
            probabilites_in = F.softmax(logits_in)
            loss = criterion(logits_in, labels)
            max_probs_in, _ = probabilites_in.max(1)

            # predict ood-probabilities
            logits_out = model.forward(ood_images)
            probabilites_out = F.softmax(logits_out)

            # calculate regularization
            loss += args.lambda1 * torch.sum((max_probs_in - A_tr) ** 2)
            loss += args.lambda2 * \
                torch.sum(torch.abs(probabilites_out -
                                    1 / train_dataset.n_classes))

            # predictions = logits.argmax(dim=1)
            # accuracy_t = torch.mean((predictions == labels).float()).item()
            if global_step % args.log_each == 0:
                # TODO: train logging for tensorboard
                # train_writer.add_scalar("Loss_BCE", loss, global_step=global_step)
                # train_writer.add_scalar("Accuracy", accuracy_t, global_step=global_step)
                # train_writer.add_scalar("Learning_rate", scheduler.get_lr()[-1], global_step=global_step)
                # log_dict_with_writer(labels, logits, train_writer, global_step=global_step)
                pass
            loss.backward()
            optimizer.step()
            global_step += 1
        print("Validating...")
        val_loss, val_acc = evaluate(
            model, val_dataloader, ood_dataloader_val, criterion, device, val_writer,
            T=args.temperature, eps=args.temperature)
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
