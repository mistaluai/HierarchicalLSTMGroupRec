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

    def __init__(self, output_dir='data', filename=None, file_id=None):
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
        self.filename = filename
        self.file_id = file_id
        self.zip_path = os.path.join(self.output_dir, filename)
        self.extract_path = os.path.join(self.output_dir, self.filename.replace(".zip", ""))
        os.makedirs(self.output_dir, exist_ok=True)

    def __download(self):
        """
        Downloads the ZIP file from Google Drive using gdown.
        """
        url = f"https://drive.google.com/uc?id={self.file_id}"
        gdown.download(url, self.zip_path, quiet=False)

    def __extract(self):
        """
        Extracts the downloaded ZIP file to the specified directory.
        """
        os.makedirs(self.extract_path, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)
        print(f"Extracted to {self.extract_path}")

    def __cleanup(self):
        """
        Deletes the ZIP file after extraction.
        """
        if os.path.exists(self.zip_path):
            os.remove(self.zip_path)
            print(f"Deleted ZIP file: {self.zip_path}")

    def __download_and_extract(self):
        """
        Downloads and extracts the specified ZIP file.
        Returns:
        --------
        str
            Path to the extracted directory.
        """
        self.__download()
        self.__extract()
        self.__cleanup()
        return self.extract_path

    def fetch_data(self):
        """
        Public method to initiate the download and extraction process.

        Returns:
        --------
        str
            Path to the extracted directory.
        """
        return self.__download_and_extract()
