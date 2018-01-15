[![Build Status](https://travis-ci.org/opengridcc/opengrid.svg?branch=master)](https://travis-ci.org/opengridcc/opengrid)
[![Coverage Status](https://coveralls.io/repos/github/opengridcc/opengrid/badge.svg?branch=master)](https://coveralls.io/github/opengridcc/opengrid?branch=master)

opengrid
========

Open-source algorithms for data-driven building analysis and control

License: Apache 2.0

# Installation
## Basic setup
For the OpenGrid code to run and to enable setup you need a working installation
of Python 3.5 (or above) and git.

You can check your Python 3.* installation by running `python3 --version` in the terminal. Git can be easily installed on most  Unix based systems by running `sudo apt-get install git-all` in the terminal.

Change your working directory to a clear location where you want to clone the opengrid library to. Then clone the repository and `cd` into it.
```
git clone https://github.com/opengridcc/opengrid.git
cd opengrid
```
(optional) set your local git username by running `git config user.name "git-username"`

## Dependencies
To be able to install all dependencies make sure you have pip3 installed. If not run `sudo apt-get install python3-pip` (linux) or `pip3 --version` (mac os)

Most of the dependencies for OpenGrid are contained in the requirements.txt file that can be found in the root directory of the cloned repository. Installing all these dependencies (and additional dependencies) can be done with the following commands:
```
sudo -H pip3 install -r requirements.txt
sudo apt-get update && sudo apt-get -y install \
   python3 \
   python3-dev \
   python3-setuptools \
   python3-pip \
   build-essential \
   libssl-dev \
   libffi-dev \
   python3-numpy \
   python3-scipy \
   python3-matplotlib \
   python3-pandas
sudo -H pip3 install jupyter
```
The last step now is to set your PYTHONPATH to the recently clone `opengrid` repository:
```
export PYTHONPATH=$PYTHONPATH:"path to chosen directory"/opengrid
```
This directory must be the root folder we just cloned.

## Testing your setup
To test if everything is setup correctly you can execute te test script:
```
// when in the root opengrid directory
cd opengrid/tests
python3 test_analyses.py
```
If the terminal prints out something like:
```
.
----------------------------------------------------------------------
Ran 1 test in 0.017s

OK
```
Then your setup is OK! Welcome to the OpenGrid development team.
