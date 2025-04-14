import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class B4Dataset(Dataset):
    def __init__(self, annotations_dataframe,split='train', transform=None, visualize=False):
        VIDEO_SPLITS = {
            'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        } 
        # Define default image transformations
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = annotations_dataframe
        self.visualize = visualize
        list_split = VIDEO_SPLITS[split]
        
        self.data = self.data.sort_values(by=['video_id', 'clip_id', 'frame_path']).reset_index(drop=True)


    def __len__(self):
        return self.data[['video_id', 'clip_id']].drop_duplicates().shape[0]
        
    def get_labels(self):
        return self.data['clip_id','clip_category'].drop_duplicates().shape[0]

    
    def __getitem__(self, idx):
        
        current_row = self.data.iloc[idx]
        video_id, clip_id = current_row['video_id'], current_row['clip_id']

        # Find all frames of this clip
        frame_paths = self.data[(self.data['video_id'] == video_id) & 
                                (self.data['clip_id'] == clip_id)]['frame_path'].tolist()


        images = [self.transform(Image.open(fp).convert('RGB')) for fp in frame_paths]
        return torch.stack(images,dtype =torch.long), current_row['clip_category']

if __name__ == "__main__":
    # Example usage
    annotations_dataframe = pd.DataFrame({
        'video_id': [1, 1, 2, 2],
        'clip_id': [1, 1, 2, 2],
        'frame_path': ['frame1.jpg', 'frame2.jpg', 'frame3.jpg', 'frame4.jpg'],
        'clip_category': ['A', 'A', 'B', 'B']
    })
    
    dataset = B4Dataset(annotations_dataframe, split='train')
      # Output the size of the images and labels
    print("Number of samples:", len(dataset))
    print("Sample images shape:", dataset[0][0].shape)  # Shape of the first sample's images
    print("Sample label:", dataset[0][1])  # Label of the first sample  
    print("Labels:", dataset.get_labels())  # Labels of the first sample