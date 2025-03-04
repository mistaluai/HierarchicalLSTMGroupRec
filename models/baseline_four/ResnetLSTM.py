import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

class ResnetEvolution(nn.Module):
    def __init__(self, b1_model, lstm_hidden_size, lstm_num_layers):
        super(ResnetEvolution, self).__init__()
        self.backbone = self.__init_backbone(b1_model)
        self.temporal = nn.LSTM(2048, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.fc = nn.Sequential() #fc yet to be implemented, input lstm hidden size, output 8 classes

    def __init_backbone(self, backbone):
        #freeze fine tuned model
        for param in backbone.parameters():
            param.requires_grad = False

        return nn.Sequential(*(list(backbone.children())[:-1]))

    def forward(self, images):
        pass