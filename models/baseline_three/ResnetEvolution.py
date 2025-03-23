import torch
import torch.nn as nn

class ResnetEvolution(nn.Module):
    def __init__(self, feature_extractor_path):
        """
        Args:
            feature_extractor (nn.Module): Pre-trained ResNet model for feature extraction.
        """
        super(ResnetEvolution, self).__init__()
        
        
        self.feature_extractor = self.__load_resnet_feature_extractor(feature_extractor_path)

        self.group_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 9)  
        )
        

    def forward(self, players):
        """
        Args:
            players: torch.Size([batch_size, num_players, 3, 224, 224])
        Returns:
            group_activity_logits: torch.Size([batch_size, 9])
        """
        batch_size, num_players, C, H, W = players.shape
        players = players.view(batch_size * num_players, C, H, W)  
        with torch.no_grad(): 
            player_features = self.feature_extractor(players) 
        player_features = player_features.view(batch_size, num_players, -1)  
        pooled_features = torch.max(player_features, dim=1)[0] 
        group_activity_logits = self.group_fc(pooled_features) 

        return group_activity_logits


    def __load_resnet_feature_extractor(state_dict_path):
        
        state_dict = torch.load(state_dict_path)
        model = FinetunableResnet()
        model.load_state_dict(state_dict, strict=False)  # strict=False in case extra keys exist
        
        feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
        
        return feature_extractor