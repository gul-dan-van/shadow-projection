import subprocess
from os import makedirs
from os.path import exists, join, basename
from types import SimpleNamespace


class ModelDownloader:
    """
    Class for handling the download of models from a cloud storage bucket.

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
        self.model_path = join(model_folder_path, f'{config.model_type.lower()}.pth')
        self.bucket_name = 'flamai'

        # Remove existing model folder if it exists
        if not exists(join(model_folder_path)):
            makedirs(join(model_folder_path))
    
    def download_models(self) -> str:
        """
        Downloads all models of a specified type from the configured cloud storage bucket.
        
        :return: The local path to the downloaded models.
        :rtype: str
        """
        print('Downloading models ...')
        command = f"wget https://storage.googleapis.com/{self.bucket_name}/{self.config.model_type.lower()}.pth -O {self.model_path}"
        process = subprocess.Popen(command, shell=True)
        process.wait()
        

    


