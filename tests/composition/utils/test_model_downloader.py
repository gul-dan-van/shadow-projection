# tests/composition/utils/test_model_downloader.py

import pytest
from src.composition.utils.model_downloader import ModelDownloader
from types import SimpleNamespace
from unittest.mock import patch, call

class TestModelDownloader:
    @pytest.fixture
    def config(self):
        return SimpleNamespace(model_type='TestModel')

    def test_download_models_positive(self, config):
        model_downloader = ModelDownloader(config, './test_models')
        with patch('subprocess.Popen') as mock_popen:
            model_downloader.download_models()
            mock_popen.assert_called_once_with(f"wget https://storage.googleapis.com/flamai/testmodel.pth -O ./test_models/testmodel.pth", shell=True)

    def test_download_models_negative(self, config):
        model_downloader = ModelDownloader(config, './test_models')
        with patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = Exception('Download failed')
            with pytest.raises(Exception):
                model_downloader.download_models()