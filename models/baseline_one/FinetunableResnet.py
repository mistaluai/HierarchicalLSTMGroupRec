import torch
import torch.nn as nn
from torchvision import models

class FinetunableResnet50(nn.Module):
    def __init__(self):
        super(FinetunableResnet50, self).__init__()
        self.model = self.__get_model()

    def __get_model(self):
        resnet = models.resnet50(weights="IMAGENET1K_V2")
        for name, param in resnet.named_parameters():
            if name.startswith("conv1") or name.startswith("layer1") or name.startswith("layer2") or name.startswith("layer3"):
                param.requires_grad = False
        num_features = resnet.fc.in_features
        layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Linear(512, 8)
        )
        resnet.fc = layers

        return resnet

    def get_fc(self):
        return self.model.fc

    def forward(self, images):
        return self.model(images)

class FinetunableResnet101(nn.Module):
    def __init__(self):
        super(FinetunableResnet101, self).__init__()
        self.model = self.__get_model()

    def __get_model(self):
        resnet = models.resnet101(weights='IMAGENET1K_V2')
        print(resnet)
        for name, param in resnet.named_parameters():
            if name.startswith("conv1") or name.startswith("layer1") or name.startswith("layer2") or name.startswith("layer3"):
                param.requires_grad = False
        num_features = resnet.fc.in_features
        layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Linear(512, 8)
        )
        resnet.fc = layers

        return resnet

    def get_fc(self):
        return self.model.fc

    def forward(self, images):
        return self.model(images)