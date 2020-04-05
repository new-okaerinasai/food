import json
import torch
import torch.nn as nn

class Config:
    def __init__(self, fpath):
        with open(fpath) as f:
            self.__dict__ = json.load(f)

def encode_onehot(labels, n_classes):
    onehot = torch.zeros(labels.size()[0], n_classes, 
                         dtype=torch.float, device=labels.device.type)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot

class GrayModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.ModuleList(list(model.children()))
        conv1 = self.features[0]
        self.features[0] = nn.Conv2d(1, conv1.out_channels, conv1.kernel_size, 
                                    conv1.stride, conv1.padding, conv1.dilation,
                                    conv1.groups, conv1.bias, conv1.padding_mode)
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

class ConfidenceNet(nn.Module):
    def __init__(self, model, num_classes):
        '''
        Class to modify neural network model to output 
        prediction vector and confidence value 
        (https://arxiv.org/pdf/1802.04865v1.pdf)

        :param model: model to modify
        :param num_classes: number of classes
        '''
        super().__init__()
        layers = list(model.children())
        if len(layers) == 1:
            layers = layers[0].children()
        self.features = nn.ModuleList(layers)        
        in_features = self.features[-1].in_features
        self.features = nn.Sequential(*self.features[:-1])
        self.classification = nn.Linear(in_features, num_classes)
        self.confidence = nn.Linear(in_features, 1)

    def forward(self, x):
        out = self.features(x).squeeze()  # [B, C, 1, 1]
        pred = self.classification(out)
        confidence = self.confidence(out)
        return pred, confidence
