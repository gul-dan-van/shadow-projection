import shutil
import subprocess

from types import SimpleNamespace
from os import makedirs
from os.path import exists, join


class ModelDownloader:
    """
    A class to handle downloading models from a cloud storage bucket.

    :param config: Configuration object containing model bucket details.
    :type config: SimpleNamespace
    :param model_folder_path: Local path to save downloaded models, defaults to './model_data'.
    :type model_folder_path: str, optional
    """

    def __init__(self, config: SimpleNamespace, model_folder_path: str = './model_data') -> None:
        """
        Initializes the ModelDownloader with configuration and model folder path.

        :param config: Configuration object containing model bucket details.
        :param model_folder_path: Local path to save downloaded models.
        """
        self.config = config
        self.model_folder_path = model_folder_path

        # Remove existing model folder if it exists
        if exists(self.model_folder_path):
            shutil.rmtree(self.model_folder_path)
    
    def download_models(self, model_type: str) -> str:
        """
        Downloads all models of a specified type from the configured cloud storage bucket.

        :param model_type: The type of models to download.
        :type model_type: str
        :return: The local path to the downloaded models.
        :rtype: str
        """
        model_path = join(self.model_folder_path, model_type)
        print('Downloading models ...')
        
        # Create model path if it does not exist
        if not exists(model_path):
            makedirs(model_path)
        
        # Construct and execute the download command
        gsutil_download_path = f"gsutil -m cp gs://{self.config.model_bucket_name}/*.* {model_path}"
        subprocess.run(gsutil_download_path, shell=True)
        return model_path
