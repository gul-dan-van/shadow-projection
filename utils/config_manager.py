"""Config Manager"""
from os.path import exists
from os import makedirs
from shutil import rmtree
from types import SimpleNamespace
from dotenv import dotenv_values


class ConfigManager:
    """
    A class to manage configuration settings for co-creation.
    """

    # Define class-level constants for folder names, default parameters, possible input types, and required keys
    FOLDERS = ['input', 'debug', 'output']
    DEFAULT_PARAMS = ['input_type', 'debug_mode']
    POSSIBLE_INPUT_TYPES = ['video', 'image', 'composite']
    REQUIRED_KEYS = {
        'image': ['foreground_image_path', 'background_image_path'],
        'video': ['foreground_video_path', 'background_video_path'],
        'composite': ['composite_frame_path', 'composite_mask_path', 'background_image_path']
    }

    MODEL_TYPE = 'PCTNet'

    def __init__(self, env_path: str = 'envs/config.env', default_config_flag: bool = False) -> None:
        self.default_config_flag = default_config_flag

        if self.default_config_flag:
            self.env_path = None

        else:
            self.env_path = env_path
        
        self.config = None

    def __read_env_file(self) -> dict:
        """
        Reads environment variables from the given file path.

        Args:
            env_path (str): The path to the environment file.

        Returns:
            dict: A dictionary containing the environment variables.
        """
        if not exists(self.env_path):
            raise ValueError(
                "Config path is incorrect, please check the path for the env file.")
        return {key.lower(): val for key, val in dotenv_values(self.env_path).items()}

    @staticmethod
    def __parse_bool(value: str) -> bool:
        """
        Parses a boolean value from a string.

        Args:
            value (str): The string value to be parsed.

        Returns:
            bool: The parsed boolean value.
        """
        return value.lower() == 'true'

    def __create_directory(self, env_var: dict) -> dict:
        """Method to create the directories listed in FOLDER list and add the path to env params"""
        if env_var is None:
            env_var = {}

        folders_to_create = ['input', 'output'] if not env_var['debug_mode'] else self.FOLDERS

        for primary_dir in folders_to_create:
            if not exists(primary_dir):
                makedirs(primary_dir)
            elif primary_dir != 'input':
                rmtree(primary_dir)
                makedirs(primary_dir)

            env_var[f'{primary_dir.lower()}_path'] = primary_dir

        return env_var

    def __validate_config(self, config: dict) -> None:
        """
        Validates the provided configuration.

        Args:
            config (dict): The configuration to be validated.
        """
        input_type = config['input_type'].lower()
        if input_type not in self.POSSIBLE_INPUT_TYPES:
            raise ValueError(f"Invalid input type: {input_type}. Must be one of: {', '.join(self.POSSIBLE_INPUT_TYPES)}.")

        for key in self.REQUIRED_KEYS[input_type]:
            if key not in config:
                raise ValueError(f"Missing required key: {key} for input type: {input_type}.")

    def generate_config(self) -> SimpleNamespace:
        """
        Gets the configuration based on the environment variables.

        Args:
            env_path (str): The path to the environment file.

        Returns:
            SimpleNamespace: A namespace object containing the configuration.
        """
        if self.default_config_flag:
            return self.generate_default_config()
        
        else:
            env_var = {}

            # Load environment variables from the specified file if it exists
            if exists(self.env_path):
                env_var.update(self.__read_env_file())

            # Parse boolean values
            env_var['model_type'] = self.MODEL_TYPE
            env_var['model_list'] = env_var['model_list'].split(',')
            env_var['debug_mode'] = self.__parse_bool(env_var['debug_mode'])
            env_var = self.__create_directory(env_var)

            # Validate configuration
            self.__validate_config(env_var)

            # Convert config to SimpleNamespace
            self.config = SimpleNamespace(**env_var)
        
    def get_config(self) -> SimpleNamespace:
        self.config = self.generate_config()
        return self.config

    def display_config(self) -> None:
        for key, value in self.config.__dict__.items():
            print(f"{key}: {value}")


    def generate_default_config(self) -> SimpleNamespace:
        """
        Generates a default configuration in SimpleNamespace format.

        Returns:
            SimpleNamespace: A namespace object containing the default configuration.
        """
        default_config = {
            'input_type': 'composite',
            'debug_mode': False,
            'model_type': 'PCTNet',
            'model_list': ['harmonization', 'border-smoothing'],
            'composite_frame_path': 'path_to_composite_frame_path',
            'composite_mask_path': 'path_to_composite_mask_path',
            'background_image_path': 'path_to_background_image_path',
        }

        self.__validate_config(default_config)  # Validate the default configuration
        return SimpleNamespace(**default_config)