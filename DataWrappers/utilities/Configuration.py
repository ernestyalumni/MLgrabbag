"""
@file Configuration.py

@brief Helper class for Python 3's configparser.
"""
from pathlib import Path
from collections import namedtuple

import configparser

class Configuration:
  """
  @class Configuration
  """

  # static variables

  # Resolve to the absolute path.
  __current_file_path = Path(__file__).resolve() 
  
  __number_of_parents_to_config_file = 1
  __number_of_parents_to_project_path = 2

  _filepath_to_config_ini = \
    __current_file_path.parents[__number_of_parents_to_config_file] / "config.ini"

  @staticmethod
  def filepath_to_config_ini():
    return Configuration._filepath_to_config_ini

  # Class methods for wrapping configparser 
  def get_configparser():
    """
    @fn get_configparser
    """
    config = configparser.ConfigParser()
    config.read(str(filepath_to_config_ini()))
    return config

  def get_quandl_API_key(self):
    """
    @fn get_quandl_API_key
    """
    return get_configparser()["Authentication"]["quandl_API_key"]
