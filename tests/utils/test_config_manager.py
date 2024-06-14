"""UNIT TEST CODE FOR CONFIG MANAGER"""
import pytest
from unittest.mock import mock_open, patch
from utils.config_manager import ConfigManager

ENV_PATH = 'envs/config_test.env'
MOCK_ENV_FILE = "DEBUG_MODE=TRUE\nINPUT_TYPE=composite\nCOMPOSITE_FRAME_PATH= path/to/composite.jpg\nCOMPOSITE_MASK_PATH= path/to/composite.jpg\nBACKGROUND_IMAGE_PATH= path/to/background.jpg\nMODEL_LIST='harmonization,border-smoothing'\nWEIGHTS_PATH=composition/image_harmonization/model/pctnet.pth\n"

@pytest.fixture(scope="class")
def config_manager():
    """
    Fixture to create an instance of ConfigManager for testing.
    
    Returns:
        ConfigManager: An instance of ConfigManager initialized with ENV_PATH.
    """

    return ConfigManager(ENV_PATH)


class TestConfigManager:
    """
    Test class for ConfigManager functionality.
    """

    @pytest.fixture(autouse=True)
    def setup_class(self, config_manager):
        """
        Fixture to set up the test class with a ConfigManager instance.
        
        Args:
            config_manager (ConfigManager): An instance of ConfigManager.
        """

        self.config_manager = config_manager

    def test_read_env_file(self):
        """
        Test method to check if __read_env_file returns a dictionary.
        """

        env_vars = self.config_manager._ConfigManager__read_env_file()
        assert isinstance(env_vars, dict)

    def test_parse_bool(self):
        """
        Test method to check the boolean parsing functionality.
        """

        assert self.config_manager._ConfigManager__parse_bool('true') is True
        assert self.config_manager._ConfigManager__parse_bool('false') is False

    def test_create_directory(self):
        """
        Test method to validate the creation of directories in the environment variables.
        """

        env_var = {'debug_mode': 'true'}
        updated_env_var = self.config_manager._ConfigManager__create_directory(env_var)
        assert 'input_path' in updated_env_var
        assert 'output_path' in updated_env_var
        assert 'debug_path' in updated_env_var

    def test_validate_config(self):
        """
        Test method to validate the configuration settings.
        """

        config = {
            'input_type': 'image',
            'foreground_image_path': 'path/to/image.jpg',
            'background_image_path': 'path/to/background.jpg'
        }
        self.config_manager._ConfigManager__validate_config(config)

    def test_get_config(self):
        """
        Test method to check if the configuration object contains required attributes.
        """

        config = self.config_manager.get_config()
        required_attrs = ['model_type', 'model_list', 'input_path', 'output_path']
        for attr in required_attrs:
            assert hasattr(config, attr)

    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_ENV_FILE)
    def test_display_config(self, mock_open, capsys):
        """
        Test method to display the configuration settings.
        """

        self.config_manager.display_config()
        captured = capsys.readouterr()
        assert "debug_mode: True" in captured.out
        assert "input_type: composite" in captured.out
        assert "composite_frame_path: path/to/composite.jpg" in captured.out
        assert "composite_mask_path: path/to/composite.jpg" in captured.out
        assert "background_image_path: path/to/background.jpg" in captured.out
        assert "model_list: ['harmonization', 'border-smoothing']" in captured.out
        assert "weights_path: composition/image_harmonization/model/pctnet.pth" in captured.out

    def test_read_env_file_exception(self):
        """
        Test method to check exception handling when reading an invalid environment file.
        """

        with pytest.raises(ValueError):
            ENV_PATH = 'invalid.env'
            config_manager = ConfigManager(ENV_PATH)
            config_manager._ConfigManager__read_env_file()

    def test_validate_config_invalid_input_type(self):
        """
        Test method to check exception handling for invalid input types in the configuration.
        """

        with pytest.raises(ValueError):
            config_manager = ConfigManager(ENV_PATH)
            config = {'input_type': 'invalid_type'}
            config_manager._ConfigManager__validate_config(config)

    def test_validate_config_missing_required_key(self):
        """
        Test method to check exception handling for missing required keys in the configuration.
        """

        with pytest.raises(ValueError):
            config_manager = ConfigManager(ENV_PATH)
            config = {'input_type': 'image'}
            config_manager._ConfigManager__validate_config(config)