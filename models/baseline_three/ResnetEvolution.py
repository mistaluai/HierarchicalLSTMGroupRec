class ResnetEvolution(nn.Module):
    def __init__(self, feature_extractor_path):
        """
        Args:
            feature_extractor (nn.Module): Pre-trained ResNet model for feature extraction.
        """
        super(ResnetEvolution, self).__init__()

        self.feature_extractor = self.__load_resnet_feature_extractor(feature_extractor_path)

        self.group_fc = nn.Sequential(
            nn.Linear(2048, 8)
        )

    def forward(self, x):
        batch_size, players, c, h, w = x.size()

        input = x.view(batch_size * players, c, h, w)
        features = self.feature_extractor(input)
        features = features.view(batch_size, players, -1)
        max_pooled = torch.max(features, dim=1)[0]
        logits = self.group_fc(max_pooled)

        return logits

    def __load_resnet_feature_extractor(self, state_dict_path):
        state_dict = torch.load(state_dict_path, weights_only=True)
        model = FinetunableResnet()
        model.load_state_dict(state_dict, strict=False)  # strict=False in case extra keys exist

        feature_extractor = nn.Sequential(*(list(model.model.children())[:-1]))

        return feature_extractor