from os.path import exists
from os import makedirs, getenv
from shutil import rmtree
from types import SimpleNamespace
from typing import Dict
from dotenv import dotenv_values

"""COCREATION MODEL CONFIGURATIONS"""
REQUIRED_MODEL_PARAMS = ['model_folder_path', 'model_list', 'features']
COCREATION_FEATURES_LIST = ['image_blending', 'harmonization', 'shadow_generation', 'border_smoothing']

DEFAULT_MODEL_CONFIG = {
    'model_folder_path': "./models",
    'model_list': ['harmonization', 'border-smoothing'],
    'features': {
        'image_blending': ['default'],
        'harmonization': ['PCTNet'],
        'shadow_generation': ['default'],
        'border_smoothing': ['default']
    }
}

class ConfigManager:
    FOLDERS = ['input', 'debug', 'output', 'models']
    DEFAULT_PARAMS = ['input_type', 'debug_mode']
    POSSIBLE_INPUT_TYPES = ['image', 'composite']
    REQUIRED_KEYS = {
        'image': ['foreground_image_path', 'background_image_path'],
        'composite': ['composite_frame_path', 'composite_mask_path', 'background_image_path'],
        'video': ['foreground_video_path', 'background_video_path']
    }

    def __init__(self, env_path: str = None) -> None:
        # Use default config or generate from env path
        self.config = self.get_default_config() if env_path is None else self.generate_config(env_path)

    @staticmethod
    def __parse_bool(value: str) -> bool:
        """
        Converts a string to a boolean.
        """
        return value.lower() == 'true'

    def __create_directory(self, env_var: dict) -> dict:
        """
        Creates directories as needed based on the environment configuration.
        """
        if env_var is None:
            env_var = {}

        # Adjust the folder list based on debug mode
        folders_without_debug = ['input', 'output', 'models']
        folders_to_create = folders_without_debug if not env_var.get('debug_mode', False) else self.FOLDERS

        # Create directories if they don't exist, avoid redundant deletion
        for primary_dir in folders_to_create:
            if not exists(primary_dir):
                makedirs(primary_dir)
            elif primary_dir != 'input' and exists(primary_dir):
                rmtree(primary_dir)
                makedirs(primary_dir)

            # Add directory paths to the environment configuration
            env_var[f'{primary_dir.lower()}_path'] = primary_dir

        return env_var

    def __read_env_file(self, env_path: str) -> dict:
        """
        Reads environment variables from a .env file.
        """
        print(env_path)
        print(exists(env_path))
        if not exists(env_path):
            raise ValueError("Config path is incorrect, please check the path for the env file.")
        return {key.lower(): val for key, val in dotenv_values(env_path).items()}

    def __validate_config(self, config: dict) -> None:
        """
        Validates the configuration to ensure all required keys are present.
        """
        input_type = config.get('input_type', '').lower()
        if input_type not in self.POSSIBLE_INPUT_TYPES:
            raise ValueError(f"Invalid input type: {input_type}. Must be one of: {', '.join(self.POSSIBLE_INPUT_TYPES)}.")

        # Check for missing keys
        missing_keys = [key for key in self.REQUIRED_KEYS[input_type] if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required key(s): {', '.join(missing_keys)} for input type: {input_type}.")

    def get_default_config(self) -> dict:
        """
        Returns the default configuration using environment variables.
        """
        # Fetch necessary environment variables only once
        env_vars = {var: getenv(var) for var in ['FOREGROUND_IMAGE_PATH', 'BACKGROUND_IMAGE_PATH', 'DEBUG_MODE', 'INPUT_TYPE']}

        # Build the default configuration dictionary
        default_config = {
            'debug_mode': False,
            'input_type': 'image',
            'model_config': DEFAULT_MODEL_CONFIG.copy(),
            'foreground_image_path': env_vars['FOREGROUND_IMAGE_PATH'],
            'background_image_path': env_vars['BACKGROUND_IMAGE_PATH'],
        }

        # Validate the default configuration
        self.__validate_config(default_config)
        return default_config

    def generate_config(self, env_path: str = None) -> SimpleNamespace:
        """
        Generates configuration based on a .env file or environment variables.
        """
        # Read environment file or build configuration from getenv
        env_params = self.__read_env_file(env_path) if env_path else {
            'debug_mode': self.__parse_bool(getenv('DEBUG_MODE', 'false')),
            'input_type': getenv('INPUT_TYPE', 'image').lower()
        }

        # Populate model configuration
        env_params['model_config'] = get_model_config(env_path, env_params)

        # Fill in required keys based on input type
        for input_type, keys in self.REQUIRED_KEYS.items():
            for key in keys:
                env_params[key.lower()] = getenv(key.upper())

        # Validate and create necessary directories
        self.__validate_config(env_params)
        self.__create_directory(env_params)

        # Return as SimpleNamespace
        return SimpleNamespace(**env_params)

    def display_config(self) -> None:
        """
        Displays the configuration settings.
        """
        print(self.config)

    def get_config(self):
        """
        Returns the current configuration.
        """
        return self.config


# Utility functions for managing model configuration
def separate_list(input_string: str) -> list:
    """
    Converts a comma-separated string into a list of words.
    """
    return input_string.replace(',', ' ').split()

def get_model_config(env_path: str, env_params: dict) -> Dict:
    """
    Generates model-related configuration based on environment settings or defaults.
    """
    if env_params is None:
        env_params = {'model_config': DEFAULT_MODEL_CONFIG.copy()}

    # If no env path, populate model config using environment variables
    if env_path is None:
        env_params['model_config'] = DEFAULT_MODEL_CONFIG.copy()
        env_params['model_config']['features'] = dict(DEFAULT_MODEL_CONFIG['features'])
        env_params['model_config']["model_folder_path"] = getenv("MODEL_FOLDER_PATH", DEFAULT_MODEL_CONFIG['model_folder_path'])
        env_params['model_config']['model_list'] = separate_list(getenv("MODEL_LIST", ','.join(DEFAULT_MODEL_CONFIG['model_list'])))

        # Fill features list from environment variables
        env_params['model_config']['features'] = {
            feature.lower(): separate_list(getenv(feature.upper(), ','.join(DEFAULT_MODEL_CONFIG['features'][feature.lower()])))
            for feature in COCREATION_FEATURES_LIST
        }

    return env_params

if __name__ == "__main__":
    ENV_PATH ="/Users/amritanshupandey/Documents/flam/image-video-blending/produciton/co-creation-pipeline/envs/config_test.env"
    config_manager = ConfigManager(ENV_PATH)
    print(config_manager.display_config())
