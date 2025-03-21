class DataProcessorBaselineTwo():

    def __init__(self, videos_root, map_classes=True):
        self.frame_mapping = {'l-spike': 0, 'l_set': 1, 'r_set': 2, 'r-pass': 3, 'r_spike': 4, 'l-pass': 5,
                              'r_winpoint': 6, 'l_winpoint': 7}
        self.player_mapping = {
            'waiting': 0, 'setting': 1, 'digging': 2, 'falling': 3, 'spiking': 4,
            'blocking': 5, 'jumping': 6, 'moving': 7, 'standing': 8}
        self.map_classes = map_classes
        self.root = videos_root

        self.data_classes, self.player_classes = self.concat_annotations()
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

    def __process_annotations(self, annotations_file):
        classes_dict = {}
        players_dict = {}
        with open(annotations_file, 'r') as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) > 1:  # Ensure there are at least 2 words
                    filename = parts[0]  # First word is the filename
                    action = parts[1]  # Second word is the action
                    classes_dict[filename] = action

                    players_annotations = self.__process_players(line)
                    players_dict[filename] = players_annotations

        return classes_dict, players_dict

    def concat_annotations(self):
        data_classes = {}
        playerdata_classes = {}
        for video_folder in sorted(os.listdir(self.root)):  # Iterate over videos
            video_path = os.path.join(self.root, video_folder)

            if not os.path.isdir(video_path):
                continue  # Skip files, process only directories

            for annotation_folder in sorted(os.listdir(video_path)):  # Iterate over clips
                annotation_path = os.path.join(video_path, annotation_folder)
                if not os.path.isdir(annotation_path):
                    if annotation_folder == 'annotations.txt':
                        frame_classes, player_classes = self.__process_annotations(annotation_path)
                        data_classes = {**frame_classes, **data_classes}
                        playerdata_classes = {**player_classes, **playerdata_classes}

        return data_classes, playerdata_classes

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
                        players_annotation = self.player_classes[frame_name]
                        dataset.append({
                            "video": video_folder,
                            "clip": clip_folder,
                            "frame": target_frame,
                            "class": self.frame_mapping[class_name],
                            "players": players_annotation
                        })
                    else:
                        missing += 1

        print(f'missing {missing} frames')
        return dataset