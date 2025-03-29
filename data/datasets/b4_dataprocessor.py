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

class DataProcessorBaselineFour:
    def __init__(self, annotation_path, videos_root):
        annotations = self._load_annotation(annotation_path)
        self.df = self._prepare_df(videos_root, annotations)

    def get_data(self):
        return self.df

    def _prepare_df(self, videos_root, annotations):
        data = []
        label_mapping = {'l-spike': 0, 'l_set': 1, 'r_set': 2, 'r-pass': 3, 'r_spike': 4, 'l-pass': 5, 'r_winpoint': 6,
                         'l_winpoint': 7}
        # Iterate through the hierarchy
        for video_id, video_data in annotations.items():
            for clip_id, clip_data in video_data.items():
                clip_category = clip_data.get('category', None)  # Activity label
                frame_boxes_dct = clip_data.get('frame_boxes_dct', {})

                for frame_id, box_infos in frame_boxes_dct.items():
                    # Append flattened data to the list
                    data.append({
                        'video_id': video_id,
                        'clip_id': clip_id,
                        'clip_category': label_mapping[clip_category],
                        'frame_path': os.path.join(videos_root + f'{video_id}/{clip_id}', (str(frame_id) + '.jpg'))
                    })

        df = pd.DataFrame(data)
        return df

    def _load_annotation(self, annotation_path):
        with open(annotation_path, 'rb') as file:
            videos_annot = pickle.load(file)
            return videos_annot

# videos_root = './volleyball/volleyball_/videos/'
# annotation_path = '../annot_all.pkl'
#
# print(DataProcessorBaselineFour(annotation_path, videos_root).get_data())
