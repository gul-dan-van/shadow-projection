from os.path import exists, join
from os import getenv
from typing import List
import requests
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # For progress bar


class ModelDownloader:
    """
    A manager class for for downloading models from GCP.

    This class provides methods to generate signed URLs, download objects, and upload objects
    to Google Cloud Storage using signed URLs.
    """

    BASE_PATH = "storage.googleapis.com"
    GCP_BUCKET_PATH = getenv("GCP_BUCKET_PATH", 'flamai')

    def __init__(self, model_folder_path: str = './models'):
        """
        Initializes the GCPStorageManager instance.

        Args:
            model_folder_path (str): The local folder path where model files will be saved.
        """
        self.model_folder_path = model_folder_path
        self.bucket_path = self.GCP_BUCKET_PATH

    def download_object(self, file: str) -> str:
        """
        Downloads an object from Google Cloud Storage with progress tracking.

        Args:
            file (str): The name of the object in GCS (without extension).

        Returns:
            str: The local file path where the object is saved.

        Raises:
            Exception: If the download or file write operation fails.
        """
        url_path = join("https://", self.BASE_PATH, self.bucket_path, f"{file}.pth")
        local_path = join(self.model_folder_path, f"{file}.pth")
        print(f"src path: {url_path}, dest_path: {local_path}")

        if exists(local_path):
            print(f"File already exists at location: {local_path}")
            return local_path

        else:
            try:
                # Use stream=True to download the file in chunks
                with requests.get(url_path, stream=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    chunk_size = 1024  # 1 KB

                    # Progress bar
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {file}", ascii=True) as pbar:
                        with open(local_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:  # Filter out keep-alive chunks
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                print(f"File downloaded successfully at location: {local_path}")
                return local_path
            except RequestException as e:
                raise Exception(f"Failed to download object: {e}")
            except IOError as e:
                raise Exception(f"Failed to write to file: {e}")

    def upload_object(self, signed_url: str, file_path: str) -> None:
        """
        Uploads an object to Google Cloud Storage using a signed URL.

        Args:
            signed_url (str): The signed URL for the object.
            file_path (str): The local file path of the object to be uploaded.

        Raises:
            Exception: If the upload or file read operation fails.
        """
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(signed_url, data=f)
                response.raise_for_status()
            print(f"File uploaded successfully: {file_path}")
        except RequestException as e:
            raise Exception(f"Failed to upload object: {e}")
        except IOError as e:
            raise Exception(f"Failed to read file: {e}")

    def download_multiple_objects(self, object_list: List[str], max_workers: int = 5) -> List[str]:
        """
        Downloads multiple objects from Google Cloud Storage concurrently.

        Args:
            object_list (List[str]): A list of object names to download.
            max_workers (int): The maximum number of threads to use for concurrent downloads.

        Returns:
            List[str]: A list of local file paths where the objects are saved.

        Raises:
            Exception: If any download or file write operation fails.
        """
        download_paths = []

        def download(file: str) -> str:
            return self.download_object(file)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(download, file): file for file in object_list}

            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    download_path = future.result()
                    download_paths.append(download_path)
                except Exception as e:
                    print(f"Failed to download {file}: {e}")

        return download_paths