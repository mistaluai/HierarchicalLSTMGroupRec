import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image



class B2Dataset(Dataset):

    VIDEO_SPLITS = {
        'train': {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54},
        'val': {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51},
        'test': {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}
    }

    def __init__(self, csv_file, split='train', transform=None):
        self.data = pd.read_csv(csv_file)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        if split in self.VIDEO_SPLITS:
            self.data = self.data[self.data['video_names'].astype(int).isin(self.VIDEO_SPLITS[split])]
        else:
            raise NameError(f'There is no such split: {split}, only {self.VIDEO_SPLITS}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['img_path']
        label = self.data.iloc[idx]['Mapped_Label']

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    


class B1Dataset(Dataset):
    VIDEO_SPLITS = {
        'train': {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54},
        'val': {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51},
        'test': {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}
    }

    def __init__(self, csv_file, tracking_annot_path, split='train', transform=None, pretrained=True, visualize=False):
        self.data = pd.read_csv(csv_file)
        self.visualize = visualize  # Flag for visualization
        self.frame_boxes_dct = load_tracking_annot(tracking_annot_path)  # Load bounding boxes per frame

        # Define default image transformations
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Filter dataset based on split
        if split in self.VIDEO_SPLITS:
            self.data = self.data[self.data['video_names'].astype(int).isin(self.VIDEO_SPLITS[split])]
        else:
            raise ValueError(f'Invalid split: {split}. Choose from {list(self.VIDEO_SPLITS.keys())}')
        
        # Initialize ResNet50 as a feature extractor (fc7 features)
        backbone = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Remove the final FC layer
        self.feature_extractor.eval()  # Set to evaluation mode
        
        # Adaptive pooling layer to combine multiple player features
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Extracts player-level features, applies pooling, and returns the frame-level representation."""
        img_path = self.data.iloc[idx]['img_path']
        label = self.data.iloc[idx]['Mapped_Label']
        frame_ID = self.data.iloc[idx]['frame_ID']  # Assuming frame_ID is available in CSV
        
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get detected players' bounding boxes for this frame
        boxes_info = self.frame_boxes_dct.get(frame_ID, [])  # List of BoxInfo objects

        # Extract and preprocess each player's crop
        preprocessed_images = []
        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box
            cropped_image = image.crop((x1, y1, x2, y2))

            # Visualization of cropped images (optional)
            if self.visualize:
                cv2.imshow('Cropped Image', np.array(cropped_image))
                cv2.waitKey(0)

            # Preprocess the cropped image
            preprocessed_images.append(self.transform(cropped_image).unsqueeze(0))

        if len(preprocessed_images) == 0:
            # If no players are detected, return a zero vector
            pooled_features = torch.zeros(2048)
        else:
            # Concatenate images into batch format: (num_players, 3, 224, 224)
            preprocessed_images = torch.cat(preprocessed_images)

            # Extract features using ResNet50
            with torch.no_grad():
                player_features = self.feature_extractor(preprocessed_images)  # (num_players, 2048, 1, 1)
                player_features = player_features.view(player_features.size(0), 2048)  # (num_players, 2048)

            # Pool features across all detected players to get a single frame representation
            player_features = player_features.unsqueeze(0).transpose(1, 2)  # Shape: (1, 2048, num_players)
            pooled_features = self.pooling(player_features).squeeze()  # Shape: (2048,)

        return pooled_features, torch.tensor(label, dtype=torch.long)


class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated