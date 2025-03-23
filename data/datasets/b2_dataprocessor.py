import os

import numpy as np


class DataProcessorBaselineTwo():

    def __init__(self, videos_root, map_classes=True):
        self.frame_mapping = {'l-spike': 0, 'l_set': 1, 'r_set': 2, 'r-pass': 3, 'r_spike': 4, 'l-pass': 5,
                              'r_winpoint': 6, 'l_winpoint': 7}
        self.player_mapping = {
            'waiting': 0, 'setting': 1, 'digging': 2, 'falling': 3, 'spiking': 4,
            'blocking': 5, 'jumping': 6, 'moving': 7, 'standing': 8}
        self.map_classes = map_classes
        self.root = videos_root

        self.dataset = self.collect_video_data()


    def get_dataset(self):
        return self.dataset

    def __process_players(self, line):
        words = line.split()
        boxes = []  # List to store (box, class) tuples

        # Skip frame ID and extra info (assuming it's always at index 1)
        words = words[2:]

        # Iterate through bounding box annotations
        for i in range(0, len(words), 5):
            x1, y1, x2, y2 = map(int, words[i:i + 4])  # Extract coordinates

            class_name = words[i + 4]  # Extract class name
            class_id = self.player_mapping[class_name]

            boxes.append(((x1, y1, x2, y2), class_id))

        return boxes

    def collect_video_data(self):
        dataset = []  # List to store video data
        videos = np.arange(55)
        for video in videos:
            video_path = os.path.join(self.root, str(video))
            annotation_path = os.path.join(video_path, 'annotations.txt')
            data_from_annotation = self.__parse_annotation(annotation_path, video_path, video)
            dataset.extend(data_from_annotation)
        return dataset

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

                # get players in frame
                players = self.__process_players(line)
                if os.path.exists(target_image_path):
                    output.append(
                        {
                            'video': video_id,
                            'frame': target_image_path,
                            'class': frame_activity_class,
                            'mapped_class': self.frame_mapping[frame_activity_class],
                            'players': players
                        }
                    )
                else:
                    print(f"Target image not found: {target_image_path}")
        return output