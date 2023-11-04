"""
@file configure_paths.py

@brief Helper functions that wrap Python 3's configparser.
"""
from collections import namedtuple
from pathlib import Path

import configparser

def _setup_paths():
    """
    @fn _setup_paths
    @brief Auxiliary function to set up configure.py, making project aware of
    its file directory paths or "position."
    """
    current_filepath = Path(__file__).resolve() # Resolve to the absolute path.

    # These values are dependent upon where this file, configure.py, is placed.
    number_of_parents_to_project_path = 2
    number_of_parents_to_config_file = 1

    filepath_to_config_ini = (
        current_filepath.parents[number_of_parents_to_config_file]
            / "config.ini")

    Setup = namedtuple(
        'Setup',
        [
            'number_of_parents_to_project_path',
            'configure_filepath',
            'config_ini_filepath'])

    return Setup(
        number_of_parents_to_project_path,
        current_filepath,
        filepath_to_config_ini)

def project_path():
    """
    @fn project_path
    @brief Returns a pathlib Path instance to this project's file path.
    """
    number_of_parents_to_project_path, current_file_path, _ = _setup_paths()
    return current_file_path.parents[number_of_parents_to_project_path]

class DataPaths:

    data_subdirectory_name = "Data"

    @staticmethod
    def list_all_files_in_directory(path):
        return list(path.iterdir())

    @staticmethod
    def get_path_with_substring(list_of_paths, input_substring = ""):
        def is_match_with_substring(path):
            return str(path).find(input_substring) != -1
        return list(filter(is_match_with_substring, list_of_paths))
