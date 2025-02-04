import os
import gdown
import zipfile


class DataFetcher:
    """
    A class to download and extract ZIP files from Google Drive.

    Attributes:
    -----------
    output_dir : str
        Directory where downloaded and extracted files will be stored.
    filename : str
        Name of the ZIP file.
    file_id : str
        Google Drive file ID for downloading.
    zip_path : str
        Full path to the downloaded ZIP file.
    extract_path : str
        Full path to the extracted contents.
    """

    def __init__(self, output_dir='data'):
        """
        Initializes the DataFetcher with output directory, filename, and Google Drive file ID.

        Parameters:
        -----------
        output_dir : str, optional
            Directory where data will be stored (default is 'data').
        filename : str
            The name of the ZIP file to be downloaded and extracted.
        file_id : str
            The Google Drive file ID of the ZIP file.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def __download(self, file_id, zip_path):
        """
        Downloads the ZIP file from Google Drive using gdown.
        """
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

    def __extract(self, zip_path, extract_path):
        """
        Extracts the downloaded ZIP file to the specified directory.
        """
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted to {extract_path}")

    def __cleanup(self, zip_path):
        """
        Deletes the ZIP file after extraction.
        """
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Deleted ZIP file: {zip_path}")

    def __download_and_extract(self, filename, file_id):
        """
        Downloads and extracts the specified ZIP file.
        Returns:
        --------
        str
            Path to the extracted directory.
        """
        zip_file = os.path.join(self.output_dir, filename)
        extract_path = os.path.join(self.output_dir, filename.replace(".zip", ""))

        self.__download(file_id, zip_file)
        self.__extract(zip_file, extract_path)
        self.__cleanup(zip_file)
        return extract_path

    def fetch_data(self, files):
        """
        Public method to initiate the download and extraction process.

        Returns:
        --------
        str
            Path to the extracted directory.
        """
        files_path = []
        for filename, file_id in files.items():
            files_path.append(self.__download_and_extract(filename, file_id))

        return files_path
