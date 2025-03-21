from torch import nn
import torchvision.models as models

class FinetunableResnet(nn.Module):
    def __init__(self):
        super(FinetunableResnet, self).__init__()
        self.model = self.__get_model()

    def __get_model(self):
        resnet = models.resnet50(weights=None)
        for name, param in resnet.named_parameters():
            if name.startswith("conv1") or name.startswith("layer1") or name.startswith("layer2"):
                param.requires_grad = False
        num_features = resnet.fc.in_features
        layers = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 9)
        )
        resnet.fc = layers

        return resnet

    def get_fc(self):
        return self.model.fc

    def forward(self, images):
        return self.model(images)