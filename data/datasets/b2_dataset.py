import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image



import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BoxInfo:
    """Parses tracking annotation lines and stores player bounding box information."""
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = (x1, y1, x2, y2)
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated


class TrackingAnnotations:
    def __init__(self, path, max_players=12):
        """
        Load tracking annotations from a file and store bounding boxes by frame.
        
        Args:
            path (str): Path to the annotation file.
            max_players (int): Maximum number of players to track (default: 12).
        """
        self.path = path
        self.max_players = max_players
        self.frame_boxes_dct = self._load_tracking_annot()

    def _load_tracking_annot(self):
        """
        Internal method to parse the tracking file and store bounding boxes by frame.
        
        Returns:
            dict: {frame_ID: list of BoxInfo objects}
        """
        player_boxes = {idx: [] for idx in range(self.max_players)}
        frame_boxes_dct = {}

        with open(self.path, 'r') as file:
            for line in file:
                box_info = BoxInfo(line)
                if box_info.player_ID >= self.max_players:
                    continue  # Ignore invalid player IDs
                
                player_boxes[box_info.player_ID].append(box_info)

        # **Process the bounding boxes**
        for player_ID, boxes_info in player_boxes.items():
            # Keep only the middle 9 frames (empirical choice)
            boxes_info = boxes_info[5:-5]  

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []
                
                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct

    def get_boxes(self, frame_ID):
        """
        Retrieve bounding boxes for a given frame.

        Args:
            frame_ID (int): The frame number.

        Returns:
            list of BoxInfo: List of player bounding boxes in this frame.
        """
        return self.frame_boxes_dct.get(frame_ID, [])

    def __len__(self):
        """ Return the number of frames with annotations. """
        return len(self.frame_boxes_dct)

    def __getitem__(self, frame_ID):
        """ Support dictionary-like access. """
        return self.get_boxes(frame_ID)


class B2Dataset(Dataset):

    VIDEO_SPLITS = {
        'train': {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54},
        'val': {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51},
        'test': {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}
    }

    def __init__(self, csv_file, tracking_annot_path, split='train', transform=None, visualize=False):
        self.data = pd.read_csv(csv_file)
        self.visualize = visualize  # Flag for visualization
        self.frame_boxes_dct = self.load_tracking_annot(tracking_annot_path)  # Load bounding boxes per frame

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
        
    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_tracking_annot(path):
        """Loads tracking annotations and returns a dictionary mapping frame_IDs to BoxInfo objects."""
        with open(path, 'r') as file:
            player_boxes = {idx: [] for idx in range(12)}
            frame_boxes_dct = {}

            for line in file:
                box_info = BoxInfo(line)
                if box_info.player_ID > 11:
                    continue
                player_boxes[box_info.player_ID].append(box_info)

            # Map from frame_ID to bounding boxes
            for player_ID, boxes_info in player_boxes.items():
                boxes_info = boxes_info[5:-6]  # Keeping the middle 9 frames
                for box_info in boxes_info:
                    if box_info.frame_ID not in frame_boxes_dct:
                        frame_boxes_dct[box_info.frame_ID] = []
                    frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct

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
                cv2.waitKey(1)  # Prevent blocking
                cv2.destroyAllWindows()

            # Apply transformation
            if self.transform:
                preprocessed_images.append(self.transform(cropped_image).unsqueeze(0))

        # Handle case where no players are detected
        if len(preprocessed_images) == 0:
            # If no players are detected, return a zero tensor with expected shape
            return torch.zeros(1, 3, 224, 224), torch.tensor(label, dtype=torch.long)

        # Concatenate images into batch format: (num_players, 3, 224, 224)
        preprocessed_images = torch.cat(preprocessed_images)

        return preprocessed_images, torch.tensor(label, dtype=torch.long)
