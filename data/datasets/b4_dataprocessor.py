import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

class B4Dataset(Dataset):
    def __init__(self, pickle_path, split='train', transform=None, visualize=False):
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
        self.data = PickleToDataFrame(pickle_path).get_dataframe()
        self.visualize = visualize
        list_split = VIDEO_SPLITS[split]
        
        self.data = self.data.sort_values(by=['video_id', 'clip_id', 'frame_path']).reset_index(drop=True)


    def __len__(self):
        self.data[['video_id', 'clip_id']].drop_duplicates().shape[0]
        
    def get_labels(self):
        return self.data['mapped_class'].tolist()
    
    def __getitem__(self, idx):
        
        current_row = self.data.iloc[idx]
        video_id, clip_id = current_row['video_id'], current_row['clip_id']

        # Find all frames of this clip
        frame_paths = self.data[(self.data['video_id'] == video_id) & 
                                (self.data['clip_id'] == clip_id)]['frame_path'].tolist()


        images = [self.transform(Image.open(fp).convert('RGB')) for fp in frame_paths]
        return torch.stack(images), current_row['clip_category']
        
        


class PickleToDataFrame:
    def __init__(self, file_path):
        """
        Initialize the class with the path to the pickle file.
        """
        self.file_path = file_path
        self.data = self.load_pickle()
        self.dataframe = self.transform_to_dataframe()

    def load_pickle(self):
        """
        Load the pickle file and store its data.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        if not self.file_path.endswith('.pkl'):
            raise ValueError("The file must be a pickle file with a .pkl extension.")
        
        with open(self.file_path, 'rb') as file:
            self.data = pickle.load(file)

    def transform_to_dataframe(self):
        """
        Transform the loaded data into a pandas DataFrame.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load the pickle file first.")
        
        self.dataframe = pd.DataFrame(self.data)

    def get_dataframe(self):
        """
        Return the transformed DataFrame.
        """
        if self.dataframe is None:
            raise ValueError("Data has not been transformed into a DataFrame yet.")
        return self.dataframe