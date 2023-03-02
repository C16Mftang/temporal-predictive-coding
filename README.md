# Predictive Coding Networks for Temporal Prediction

Repository for experiments with the temporal predictive coding model

## 1. Description
This repository contains code to perform experiments with temporal predictive coding models.


## 2. Installation
To run the code, you should first install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html) (preferably the latter), 
and then clone this repository to your local machine.

Once these are installed and cloned, you can use the appropriate `.yml` file to create a conda environment. 
For Ubuntu or Mac OS, open a terminal (for Windows, open the Anaconda Prompt), go to the directory where you cloned the repo and then enter:

1. `cd temporal-predictive-coding`
2. `conda env create -f environment.yml`  
3. `conda activate temporalenv`
4. `pip install -e .`  

## 3. Use
Once the above are done, you can reproduce figures from the paper:

For Figure 3 enter:

`python scripts/tracking_inf_steps.py` (panel A, B, C) and

`python scripts/tracking_inf_multi_seeds.py` (panel D)

For Figure 4 enter:

`python scripts/tracking_learning_AC.py`

Once you run these commands, a directory named `results` will the be created to store all the data and figures collected from the experiments.