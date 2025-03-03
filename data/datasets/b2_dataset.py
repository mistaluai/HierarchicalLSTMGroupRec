import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data.datasets.b2_dataprocessor import DataProcessorBaselineTwo
class B2Dataset(Dataset):

    VIDEO_SPLITS = {
        'train': {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54},
        'val': {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51},
        'test': {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}
    }

    def __init__(self, data, split='train', transform=None, player_transform = None , visualize=False):

        # Define default image transformations
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.player_transform = player_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.data = data
        
    def __len__(self):
        return {
            'frames': len(self.data),
            'players': sum([len(boxes) for boxes in self.data['boxes']])
            }   
        
    def __getitem__(self, idx):
        """Extracts player-level features, applies pooling, and returns the frame-level representation."""
        item = self.data[idx]
        frame = Image.open(item['frame']).convert("RGB")
        frame_class = item['class']
        
        if self.transform:
            frame = self.transform(frame)
        
        # Load and transform player bounding boxes
        player_images = []
        player_labels = []
        for (bbox, player_label) in item["players"]:
            x1, y1, x2, y2 = bbox
            player_image = frame.crop((x1, y1, x2, y2))
            if self.player_transform:
                player_image = self.player_transform(player_image)
            player_images.append(player_image)
            player_labels.append(player_label)

        return frame, frame_class, player_images, player_labels
            
    
        
        

        
    def preprocess_crops(self, image, boxes_info):
        """
        Extracts and preprocesses cropped player images from a given frame.
        """
        preprocessed_images = []
        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box
            cropped_image = image.crop((x1, y1, x2, y2))

            # Visualization (Optional)
            if self.visualize:
                cv2.imshow('Cropped Image', np.array(cropped_image))
                cv2.waitKey(1)
                cv2.destroyAllWindows()

            # Apply transformations
            if self.transform:
                preprocessed_images.append(self.transform(cropped_image).unsqueeze(0))

        # If no players detected, return a zero tensor
        if len(preprocessed_images) == 0:
            return torch.zeros(1, 3, 224, 224)

        return torch.cat(preprocessed_images)  # Shape: (num_players, 3, 224, 224)

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Extracts player-level features, applies pooling, and returns the frame-level representation."""
        img_path = self.data.iloc[idx]['img_path']
        label = self.data.iloc[idx]['Mapped_Label']
        frame_ID = self.data.iloc[idx]['frame_ID']
        
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get bounding boxes
        boxes_info = self.tracking_annot.get_boxes(frame_ID)

        # Preprocess cropped player images
        preprocessed_images = self.preprocess_crops(image, boxes_info)
        return preprocessed_images, torch.tensor(label, dtype=torch.long)
    
### Custom Collate Function to Handle Variable Player Counts Need to be transfered to the utils files
def custom_collate_fn(batch):
    """
    Custom collate function for batching frames with a variable number of detected players.

    Args:
        batch: List of tuples [(tensor of player crops, label), ...]

    Returns:
        images_list: List of lists of player images (each batch has variable-length lists)
        labels: Tensor of shape (batch_size,)
    """
    images_list = [sample[0] for sample in batch]  # List of cropped player tensors
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)

    return images_list, labels

