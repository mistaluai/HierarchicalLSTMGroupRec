import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import torchvision.models as models

class ResnetEvolution(nn.Module):
    def __init__(self, input_size=2048, lstm_hidden_size=512, num_frames=9, num_classes=8, model_state_dict=None, backbone=models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=False)):
        super(ResnetEvolution, self).__init__()
        self.feature_extractor = self.__init_backbone(backbone, model_state_dict)

        self.temporal = nn.LSTM(input_size, lstm_hidden_size, num_frames, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def __init_backbone(self, backbone, model_state_dict):
        if model_state_dict is not None:
            backbone.load_state_dict(model_state_dict)
            backbone = nn.Sequential(*list(backbone.model.children())[:-1])

            for param in backbone.parameters():
                param.requires_grad = False
        else:
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        return backbone

    def forward(self, images):
        batch_size, seq, c, h, w = images.shape

        images_reshaped = images.reshape(batch_size * seq, c, h, w)
        features = self.feature_extractor(images_reshaped) #shape is (batch * 9, 2048, 1, 1)
        features = features.reshape(batch_size, seq, -1) # shape is (batch, 9, 2048)

        temporal_features, _ = self.temporal(features) #output is (batch, 9 , hidden_size)
        last_feature = temporal_features[:, -1, :]

        logits = self.fc(last_feature)

        return logits

