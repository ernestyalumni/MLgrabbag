"""
@file drive.py
@brief Really, this is a test file to use config.ini and configparser,
experimenting with different file directories, folder structures.
@ref https://github.com/nasa/podaacpy/blob/165b71250008c70ac7b30e5663339bf8e903d992/podaac/drive.py
"""

import configparser
import os
from pathlib import Path

print('' == True)
print('' == False)
print("" == True)
print("" == False)

empty_char = ''
empty_string = ""

if (empty_char):
  print("An empty_char is true")
else:
  print("An empty_char is false")


if (empty_string):
  print("An empty_string is true")
else:
  print("An empty_string is false")


# @ref https://docs.python.org/3/library/configparser.html

config = configparser.ConfigParser()

# @ref https://docs.python.org/3/library/configparser.html
# Attempt to read and parse an iterable of filenames, returning a list of
# filenames which were successfully parsed.
#
# If filenames is a string, a bytes object or a path-like object, it's treated
# as a single filename. If a file named in filenames cannot be opened, that file
# will be ignored. This is designed so that you can specify an iterable of
# potential configuration file locations (for example, the current directory,
# the user's home directory, and some system-wide directory), and all existing
# configuration files in the iterable will be read.

os.path.realpath(__file__)
print("This is os.path's realpath :", os.path.realpath(__file__))

# @url https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
# Path.resolve(strict=False)
# Make the path absolute, resolving any symlinks. A new path object is
# returned.

file_path = Path(__file__)
#print(file_path.as_uri())
file_path = file_path.resolve()
print("This is an absolute file URI: ", file_path.as_uri())

print(file_path.parents[0].as_uri())
print(file_path.parents[1].as_uri())
print(file_path.parents[2].as_uri())


print("All path objects of the directory parent #2 contents: ")
for child in file_path.parents[2].iterdir():
  print(child)

# @url https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
# @brief Slash operator helps create child paths, similar to os.path.join()
print(file_path.parents[2] / "config.ini")
print(type(file_path.parents[2] / "config.ini"))

filepath_to_config_ini = file_path.parents[2] / "config.ini"

print("Filepath to config ini as a string")
print(str(filepath_to_config_ini))

# THIS WORKS
#config.read((file_path.parents[2] / "config.ini").as_uri())

# THIS WORKS
config.read(str(filepath_to_config_ini))

# THIS WORKS
#config.read_file(open(str(filepath_to_config_ini), 'r'))

print(config.sections())
print(type(config["paths"]))
print(dir(config["paths"]))
print(config["paths"].keys())
print(config["paths"]["data_folder_absolute_path"])

data_path = Path(config["paths"]["data_folder_absolute_path"])