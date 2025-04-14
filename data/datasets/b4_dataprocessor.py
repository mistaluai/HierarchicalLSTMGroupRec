import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from data.datasets.b4_dataset import B4Dataset


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
# df = DataProcessorBaselineFour(annotation_path, videos_root).get_data()
#
# dataset = B4Dataset(annotations_dataframe=df)
# print(dataset.__len__())
