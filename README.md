# Formality Evaluation

This repository includes work on the formality evaluation task. 

## Content

* `report.md` report on the work done with description of actions taken.
* `scripts` scripts for data preprocessing and evaluation
    * `preprocess_datasets.py` script for preprocessing dataset. It s required to set two variables in the begining and download german dataset
    * `evaluate_model.py` model evaluation script

##  Usage

* create python environment
* `pip install -r requirements.txt`
* download German dataset
* `cd scripts`
* `python preprocess_datasets.py`
* `python evaluate_model.py -d <path to dataset> -m <model name> -method <"classification" | "spearman" | "both">`