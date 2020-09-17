""" Configuration file for pytest """

import pytest
from pathlib import Path
from sys import path

ROOT = Path(__file__).parent.parent
path.append(ROOT)