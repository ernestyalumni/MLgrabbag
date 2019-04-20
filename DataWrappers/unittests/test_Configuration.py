"""
@file test_Configuration.py
"""
import pytest

from quandl_wrapper.utilities.Configuration import Configuration

@pytest.fixture
def config_ini_filepath():
  """
  @fn config_ini_filepath
  """
  import os
  return os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini")

def test_filepath_to_config_ini(config_ini_filepath):
  """
  @fn test_filepath_to_config_ini
  """
  configuration = Configuration()
  assert str(configuration.filepath_to_config_ini()) == \
    config_ini_filepath

