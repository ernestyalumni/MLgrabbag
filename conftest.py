"""
@file conftest.py

@details Guard access to resources with fixtures.

From
https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
this makes it so that pytest looks for the conftest and pytest adds the parent
directory of conftest.py to the sys.path (in this case, /Voltron).

No need to write custom code for mangling sys.path or remember to drag
PYTHONPATH along, or placing __init__.py into dirs where it doesn't belong.
"""

import pytest
import requests