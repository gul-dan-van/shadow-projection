"""Condig Manager"""
from os.path import exists
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

    def __init__(self) -> None:
        pass

    @staticmethod
    def __read_env_file(env_path: str) -> dict:
        """
        Reads environment variables from the given file path.

        Args:
            env_path (str): The path to the environment file.

        Returns:
            dict: A dictionary containing the environment variables.
        """
        if not exists(env_path):
            raise ValueError(
                "Config path is incorrect, please check the path for the env file.")
        return {key.lower(): val for key, val in dotenv_values(env_path).items()}

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
                raise ValueError(f"Missing required key: {
                                 key} for input type: {input_type}.")

    def get_config(self, env_path: str) -> SimpleNamespace:
        """
        Gets the configuration based on the environment variables.

        Args:
            env_path (str): The path to the environment file.

        Returns:
            SimpleNamespace: A namespace object containing the configuration.
        """
        env_var = {}

        # Load environment variables from the specified file if it exists
        if exists(env_path):
            env_var.update(self.__read_env_file(env_path))

        # Parse boolean values
        env_var['model_type'] = self.MODEL_TYPE
        env_var['model_list'] = env_var['model_list'].split(',')
        env_var['debug_mode'] = self.__parse_bool(env_var['debug_mode'])

        # Validate configuration
        self.__validate_config(env_var)

        # Convert config to SimpleNamespace
        return SimpleNamespace(**env_var)
