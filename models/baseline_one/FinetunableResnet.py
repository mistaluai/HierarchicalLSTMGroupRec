import torch
import torch.nn as nn
from torchvision import models

## this is the final model used in the b1
class FinetunableResnet(nn.Module):
    def __init__(self):
        super(FinetunableResnet, self).__init__()
        self.model = self.__get_model()

    def __get_model(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = resnet.fc.in_features
        layers = nn.Sequential(
            nn.Linear(num_features, 8)
        )
        resnet.fc = layers

        return resnet

    def get_fc(self):
        return self.model.fc

    def forward(self, images):
        return self.model(images)