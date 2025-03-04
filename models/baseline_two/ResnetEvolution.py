import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


class ResnetEvolution(nn.Module):
    def __init__(self):
        super(ResnetEvolution, self).__init__()
        self.model, backbone_outputs = self.__init_backbone(torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))

        self.layers = nn.Sequential(
            nn.Linear(backbone_outputs, 4096),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 9)
        )

    def __init_backbone(self, backbone):
        num_features = backbone.fc.in_features
        backbone.fc = None
        return backbone, num_features

    def get_fc(self):
        return self.layers

    def forward(self, images):
        features = self.model(images)
        logits = self.layers(features)
        return logits