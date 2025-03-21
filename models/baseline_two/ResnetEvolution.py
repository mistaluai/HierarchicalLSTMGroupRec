class ResnetEvolution(nn.Module):
    def __init__(self):
        super(ResnetEvolution, self).__init__()
        self.model = self.__init_backbone(torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=False))

    def __init_backbone(self, backbone):
        num_features = backbone.fc.in_features
        layers = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 9)
        )
        backbone.fc = layers
        return backbone

    def get_fc(self):
        return self.model.fc

    def forward(self, images):
        return self.model(images)