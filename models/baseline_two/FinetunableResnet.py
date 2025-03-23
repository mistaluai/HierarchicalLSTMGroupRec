from torch import nn
import torchvision.models as models

class FinetunableResnet(nn.Module):
    def __init__(self):
        super(FinetunableResnet, self).__init__()
        self.model = self.__init_backbone(torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=False))

    def __init_backbone(self, backbone):
        num_features = backbone.fc.in_features
        layers = nn.Sequential(
            nn.Linear(num_features, 9),
        )
        backbone.fc = layers
        return backbone

    def get_fc(self):
        return self.model.fc

    def forward(self, images):
        return self.model(images)