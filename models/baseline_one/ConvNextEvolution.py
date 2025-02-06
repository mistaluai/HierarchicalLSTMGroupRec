from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

class ConvNextEvolution(nn.Module):
    def __init__(self, hidden_layers=[]):
        super(ConvNextEvolution, self).__init__()
        self.hidden_layers = hidden_layers
        self.model = self.__init_backbone(models.convnext_base(pretrained=True))

    def __init_backbone(self, backbone):
        num_features = backbone.classifier[2].in_features

        layers = []
        input_size = num_features  # Start with backbone output size
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # Activation function
            input_size = hidden_size  # Update input for next layer

        layers.append(nn.Linear(input_size, 8))  # Final output layer

        backbone.classifier[2] = nn.Sequential(*layers)  # Output layer for binary classification

        return backbone

    def get_fc(self):
        return self.model.classifier[2]

    def forward(self, images):
        return self.model(images)