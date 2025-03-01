import os

import pandas as pd
from charset_normalizer.md import annotations


class DataProcessorBaselineOne():
    def __init__(self, videos_root, map_classes=True):
        self.root = videos_root
        self.data_classes = self.concat_annotations()
        self.dataset = self.collect_video_data()

        self.map_classes = map_classes
        self.label_mapping = {'l-spike': 0, 'l_set': 1, 'r_set': 2, 'r-pass': 3, 'r_spike': 4, 'l-pass': 5, 'r_winpoint': 6, 'l_winpoint': 7}

        self.dataset_df = self.__prepare_df()


    def __prepare_df(self):
        df = pd.DataFrame(self.dataset)
        if self.map_classes:
            df['mapped_class'] = df['class'].map(self.label_mapping)
        return df

    def get_dataset(self):
        return self.dataset

    def get_dataset_df(self):
        return self.dataset_df

    def __process_annotations(self, annotations_file):
        data_dict = {}

        with open(annotations_file, 'r') as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) > 1:  # Ensure there are at least 2 words
                    filename = parts[0]  # First word is the filename
                    action = parts[1]  # Second word is the action
                    data_dict[filename] = action

        return data_dict

    def concat_annotations(self):
        data_classes = {}
        for video_folder in sorted(os.listdir(self.root)):  # Iterate over videos
            video_path = os.path.join(self.root, video_folder)

            if not os.path.isdir(video_path):
                continue  # Skip files, process only directories

            for annotation_folder in sorted(os.listdir(video_path)):  # Iterate over clips
                annotation_path = os.path.join(video_path, annotation_folder)
                if not os.path.isdir(annotation_path):
                    if annotation_folder == 'annotations.txt':
                        data_classes = {**self.__process_annotations(annotation_path), **data_classes}

        return data_classes

    def collect_video_data(self):
        dataset = []  # List to store video data
        missing = 0
        for video_folder in sorted(os.listdir(self.root)):  # Iterate over videos
            video_path = os.path.join(self.root, video_folder)
            if not os.path.isdir(video_path):
                continue  # Skip files, process only directories

            for clip_folder in sorted(os.listdir(video_path)):  # Iterate over clips
                clip_path = os.path.join(video_path, clip_folder)
                if not os.path.isdir(clip_path):
                    continue

                frames = sorted(
                    [os.path.join(clip_path, f) for f in os.listdir(clip_path) if f.endswith(".jpg")]
                )
                if len(frames) == 41:  # Ensure expected number of frames
                    target_frame = frames[20]
                    frame_name = target_frame.split('/')[-1]
                    if frame_name in self.data_classes:
                        class_name = self.data_classes[frame_name]
                        dataset.append({
                            "video": video_folder,
                            "clip": clip_folder,
                            "frame": target_frame,
                            "class": class_name
                        })
                    else:
                        missing += 1

        print(f'missing {missing} frames')
        return dataset