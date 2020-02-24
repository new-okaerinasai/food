import torch
from torchvision.transforms import ToTensor
import argparse
from datasets import CIFAR100, TinyImagenet

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['tiny_imagenet', 'cifar100'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    if args.dataset == 'tiny_imagenet':
        train_dataset = TinyImagenet(args.data_path, mode="train")
        test_dataset = TinyImagenet(args.data_path, mode="test")
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(args.data_path, train=True, transform=ToTensor())
        test_dataset = CIFAR100(args.data_path, train=False, transform=ToTensor())

if __name__ == '__main__':
    train()
