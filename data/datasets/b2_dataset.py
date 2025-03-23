import random

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


class B3Dataset(Dataset):
    VIDEO_SPLITS = {
        'train': {1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54},
        'val': {0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51},
        'test': {4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47}
    }

    def __init__(self, data, split='train', transform=None, player_transform=None, visualize=False):

        # Define default image transformations
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.player_transform = player_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        frame = Image.open(item['frame']).convert("RGB")
        frame_class = item['mapped_class']


        # Load and transform player bounding boxes
        player_images = []
        player_labels = []
        for (bbox, player_label) in item["players"]:
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            player_image = frame.crop((x1, y1, x2, y2))
            player_image = self.player_transform(player_image)
            player_images.append(player_image)
            player_labels.append(player_label)

        #convert to tensors
        player_images = torch.stack(player_images)
        player_labels = torch.tensor(player_labels, dtype=torch.long)

        frame = self.transform(frame)

        return frame, frame_class, player_images, player_labels


class PlayerDataset(Dataset):
    def __init__(self, dataset, split='train', transform=None, downsampled_class=8, downsample_ratio=0.15, downsample=True):
        self.dataset = dataset
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        VIDEO_SPLITS = {
            'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }

        self.index_map = []
        list_split = VIDEO_SPLITS[split]
        self.labels = []

        downsampled_class_index_map = []
        for item_idx, item in enumerate(self.dataset):
            if int(item['video']) in list_split:
                for player_idx, (bbox, action_class) in enumerate(item['players']):
                    if action_class != downsampled_class:
                        self.index_map.append((item_idx, player_idx))
                        self.labels.append(action_class)
                    else:
                        downsampled_class_index_map.append((item_idx, player_idx))

        downsample_size = int(len(downsampled_class_index_map) * downsample_ratio)
        if downsample:
            downsampled_class_index_map = random.sample(downsampled_class_index_map, downsample_size)

        self.index_map.extend(downsampled_class_index_map)
        self.labels.extend([downsampled_class] * len(downsampled_class_index_map))

        print(f'the {split} has {len(self.index_map)} samples')
        self.invalid = 0

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        item_idx, player_idx = self.index_map[idx]
        item = self.dataset[item_idx]
        frame_path = item['frame']
        bbox, action_class = item['players'][player_idx]

        image = Image.open(frame_path).convert('RGB')
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        bbox = (x1, y1, x2, y2)

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            self.invalid += 1
            print(f'invalids:{self.invalid}')
            return torch.rand(3, 224, 244), torch.tensor(action_class, dtype=torch.long)

        cropped_image = image.crop(bbox)  # (x1, y1, x2, y2)

        cropped_image = self.transform(cropped_image)

        return cropped_image, torch.tensor(action_class, dtype=torch.long)

