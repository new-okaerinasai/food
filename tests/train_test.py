import torch
import sys
sys.path += ["..", "../experiments"]
from experiments.train_vanilla import evaluate, train
from food.datasets import TinyImagenet, MNIST, FashionMNIST, CIFAR_10, CIFAR_100
import food


def test_train():
    datasets = ['tiny_imagenet', 'cifar_100', 'cifar_10']
    models = ['resnet18', 'resnet50']
    for model in models:
        for dataset in datasets:
            batch_size = 32
            logits, loss, predictions, val_loss, val_acc = train(model=model, dataset=dataset,
                                                                 epochs=1, test=True, batch_size=batch_size,
                                                                 config="../experiments/vanilla.json")

            assert logits.shape[0] == batch_size
            assert len(predictions) == batch_size
            assert loss > 0
            assert val_loss >= 0
            assert val_acc > 0
