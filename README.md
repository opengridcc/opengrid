[![Build Status](https://travis-ci.org/EpoxyD/opengrid.svg?branch=master)](https://travis-ci.org/EpoxyD/opengrid)
[![Coverage Status](https://coveralls.io/repos/github/EpoxyD/opengrid/badge.svg?branch=master)](https://coveralls.io/github/EpoxyD/opengrid?branch=master)

# opengrid

Open-source algorithms for data-driven building analysis and control

License: Apache 2.0

# Installation

## Basic setup

For the OpenGrid code to run and to enable setup you need a working installation of Python 3.5 (or above) and git.

You can check your Python 3.\* installation by running `python3 --version` in the terminal. Git can be easily installed on most Unix based systems by running `sudo apt-get install git-all` in the terminal.

Change your working directory to a clear location where you want to clone the opengrid library to. Then clone the repository and `cd` into it.

```
git clone https://github.com/opengridcc/opengrid.git
cd opengrid
```

## Dependencies

To be able to install all dependencies make sure you have pip3 installed. If not run `sudo apt-get install python3-pip` (linux) or `pip3 --version` (mac os)

### Virtual environments

For safety and clarity we'll use a virtuel environment for development. Start a new one and activate it before moving on. 

``` bash
python -m venv venv
source ./venv/bin/activate
```

When you are done developing, do not forget to close down the virtual environment. Do not remove the test environment if you are planning to use it again later. You can just re-activate it.

``` bash
deactivate
(optional) rm -rf venv
```

### Test Dependencies

Install test dependencies inside of your virtual environment (after activation).

```python
python3 -m pip install -r requirements.txt
```

### Package dependencies

Install the package dependencies by installing the package locally

``` python
python3 -m pip install .
```

## Testing your setup

To test if everything is setup correctly you can execute the test scripts:

``` bash
python3 -m pytest test/analysis.py
```

If the terminal prints out something like:

``` bash
test/test_analyses.py::AnalysisTest::test_count_peaks PASSED               [ 16%]
test/test_analyses.py::AnalysisTest::test_load_factor PASSED               [ 33%]
test/test_analyses.py::AnalysisTest::test_share_of_standby_1 PASSED        [ 50%]
test/test_analyses.py::AnalysisTest::test_share_of_standby_2 PASSED        [ 66%]
test/test_analyses.py::AnalysisTest::test_standby PASSED                   [ 83%]
test/test_analyses.py::AnalysisTest::test_standby_with_time_window PASSED  [100%]
```

Then your setup is OK! Welcome to the OpenGrid development team.
