# Car Regcognition

## Introduction

## Approach

## Usage
### Setup environment
This project is using python 3.6.8 & virtualenv to create virtual environment name (ai4c).
```bash
$ sudo apt-get install python3-pip
$ sudo pip3 install virtualenv
$ virtualenv ai4c
$ source ai4c/bin/activate
```
If you are using other python version (>3.6), please be noted to update the IM2REC_PY_PATH. 
```
im2rec_path = "ai4c/lib/python3.6/site-packages/mxnet/tools/im2rec.py"
```

### Install required packages
Once ai4c environment is activated, installed required packages as listed in requirement.txt
It is important to use same version of packages to prevent unexpected incompatible during running script.
```bash
pip install -r requirements.txt
```

Extract the car_ims dataset to the project folder.
 
