# opengrid

Open-source algorithms for data-driven building analysis and control

[![License](https://img.shields.io/github/license/EpoxyD/opengrid)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://travis-ci.org/EpoxyD/opengrid.svg?branch=master)](https://travis-ci.org/EpoxyD/opengrid)
[![Coverage Status](https://coveralls.io/repos/github/EpoxyD/opengrid/badge.svg?branch=master)](https://coveralls.io/github/EpoxyD/opengrid?branch=master)

## Installation

### Local

Install the current local opengrid files using the following command:

``` bash
python3 -m pip install .
```

### Latest packaged version

Install the latest available published opengrid package using

``` bash
python3 -m pip install opengrid
```

## Development

### Basics

Make sure the following are installed on your system:

|         | Linux | MacOS | Windows |
|---------|-------|-------|---------|
| git     | ``` apt install git ``` | ``` brew install git ``` | [via git-scm](https://git-scm.com/download) |
| python3 | ``` apt install python3 ``` | ``` brew install python3 ``` | [via python.org](https://www.python.org/downloads/windows/) |

Check if installation was succesful by running:

``` bash
git --version
python3 --version
```

Clone the opengrid repository to wherever you like:

``` bash
git clone https://github.com/EpoxyD/opengrid.git && cd opengrid
```

### [OPTIONAL] Virtual environment

For safety and clarity we suggest you use a virtual development environment. This environment makes sure that there is no ambiguity between your already available python packages on your system and the ones you'll be using for development.

Get started using the following command. You are inside the new environment once you see the environment name in front of your terminal entry.

``` bash
python3 -m venv venv
source ./venv/bin/activate
```

When you are done developing, deactivate the virtual environment. This will give you back access to your own python packages. Do not remove the test environment if you are planning to use it again later. You can just re-activate it.

``` bash
deactivate
(optional) rm -rf venv
```

### Development required packages

Install required packages for development (Inside of virtual environment if you are using this).

``` bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r test/requirements.txt
```

## Testing your setup

First, install the local package ([as mentioned above](#Local)). This will install the opengrid dependencies on your system, or in your virtual environment.

To test if everything is setup correctly you can execute one of the test scripts:

``` bash
python3 -m pytest test/test_analyses.py
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
