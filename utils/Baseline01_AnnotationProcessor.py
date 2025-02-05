import os
import glob
import pandas as pd

class AnnotationProcessor:
    def __init__(self, base_path, output_path='/kaggle/working/', filename='dataset.csv'):
        self.base_path = base_path
        self.output_path = output_path
        self.filename = filename
        self.data = None
        self.run()

    def get_group_annotation(self, file, folder_name):
        """Extract the first two elements from each row of the annotation file."""
        with open(file, 'r') as f:
            data = [line.split()[:2] for line in f]

        df = pd.DataFrame(data, columns=['FrameID', 'Label'])
        df['video_names'] = folder_name
        # Ensure the output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        # Save the file directly in the root of the output path
        df.to_csv(os.path.join(self.output_path, f'{folder_name}.csv'), index=False)

    def process_annotations(self):
        """Process annotations from all folders in the base path."""
        for folder_name in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, folder_name)
            if os.path.isdir(folder_path):
                annotated_file_path = os.path.join(folder_path, 'annotations.txt')
                self.get_group_annotation(annotated_file_path, folder_name)

    def combine_csv_files(self):
        """Combine all CSV files into a single DataFrame."""
        csv_files = glob.glob(os.path.join(self.output_path, '*.csv'))
        data = [pd.read_csv(csv_file) for csv_file in csv_files]
        return pd.concat(data, ignore_index=True)

    def generate_img_paths(self, df):
        """Generate image paths based on the DataFrame."""
        df['img_path'] = df.apply(
            lambda x: os.path.join(
                self.base_path,
                str(x['video_names']),  # Ensure `video_names` is a string
                str(x['FrameID'])[:-4],  # Ensure `FrameID` is a string and remove the last 4 characters
                str(x['FrameID'])  # Ensure `FrameID` is a string
            ), axis=1
        )
        return df

    def save_combined_data(self, df):
        """Save the combined data to a CSV file."""
        # Ensure the output directory exists before saving
        os.makedirs(self.output_path, exist_ok=True)
        # Save the combined data directly in the root of the output path
        df.to_csv(os.path.join(self.output_path, self.filename), index=False)

    def cleanup(self):
        """Delete all intermediate CSV files except the final output file."""
        for csv_file in glob.glob(os.path.join(self.output_path, '*.csv')):
            if not csv_file.endswith(self.filename):
                os.remove(csv_file)

    def run(self):
        """Run the whole annotation processing pipeline."""
        self.process_annotations()
        combined_data = self.combine_csv_files()
        data_with_paths = self.generate_img_paths(combined_data)
        self.save_combined_data(data_with_paths)
        self.cleanup()  # Clean up intermediate files
        self.data = pd.read_csv(os.path.join(self.output_path, self.filename))