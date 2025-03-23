import os

import pandas as pd
from charset_normalizer.md import annotations
import numpy as np

class DataProcessorBaselineOne():
    def __init__(self, videos_root):
        self.root = videos_root
        self.label_mapping = {'l-spike': 0, 'l_set': 1, 'r_set': 2, 'r-pass': 3, 'r_spike': 4, 'l-pass': 5, 'r_winpoint': 6, 'l_winpoint': 7}
        self.dataset = self.collect_video_data()


    def get_dataset_df(self):
        return pd.DataFrame(self.dataset)

    def get_dataset(self):
        return self.dataset

    def __parse_annotation(self, annotation, video_path, video_id):
        output = []
        with open(annotation, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Extract the frame ID (image name) and frame activity class
                frame_image = parts[0]  # Image name (e.g., '48075.jpg')
                frame_activity_class = parts[1]  # Frame Activity Class (e.g., 'r_winpoint')

                # Remove .jpg from frame_image to get the frame ID
                frame_id = os.path.splitext(frame_image)[0]

                # Construct the expected path to the target image
                target_image_path = os.path.join(video_path, str(frame_id), frame_image)

                if os.path.exists(target_image_path):
                    output.append(
                        {
                        'video': video_id,
                        'frame': target_image_path,
                        'class': frame_activity_class,
                        'mapped_class': self.label_mapping[frame_activity_class]
                        }
                    )
                else:
                    print(f"Target image not found: {target_image_path}")
        return output

    def collect_video_data(self):
        dataset = []  # List to store video data
        videos = np.arange(55)
        for video in videos:
            video_path = os.path.join(self.root, str(video))
            annotation_path = os.path.join(video_path, 'annotations.txt')
            images = self.__parse_annotation(annotation_path, video_path, video)
            dataset.extend(images)
        return dataset