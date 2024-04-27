# Overview

This is the offical codebase for the Paper _Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models_

## Abstract
Abstract here

# Quick Start (Running the Code)



The following subsections give examples commands to run the training, evaluations and analysis necessary to reproduce the results in our paper.

In the paper, `tiny`, `base`, `small` and `medium` refer to the multi-lingual versions of the Whisper models, whilst `tiny.en`, `base.en`, `small.en` and `medium.en` refer to the English-only versions of the Whisper Models (this is the nomenclature used in the original Whisper paper). However, we use a slightly different naming convention in the codebase when specifying the model name as an argument to the scripts. The Table below gives the mapping from the names used in the paper and the equivalent names used in the codebase.

| Model name in paper | `model_name` in codebase |
| --------------- | ------------------- |
| tiny.en | whisper-tiny |
| tiny | whisper-tiny-multi |
| base.en | whisper-base |
| base | whisper-base-multi |
| small.en | whisper-small |
| small | whisper-small-multi |
| medium.en | whisper-medium |
| medium | whisper-medium-multi |


You can see all the arguments used in the different scripts in `src/tools/args.py`.

## Package Installation

This code has been tested on python>=3.9.

Fork the repository and then git clone

`git clone https://github.com/rainavyas/prepend_acoustic_attack`


Install all necessary packages by creating a conda environment from the existing `environment.yml` file.

```
conda env create -f environment.yml
conda activate venv_gector
```


## Learning the universal prepend acoustic attack

## Evaluating a universal prepend acoustic attack

### Transfer attack evaluation

## Analysis


# Data

Describe here how certain datasets need to pre-downloaded and stored in particular directory. Also specify which part of the code needs to be updated to point to this directory.

# Results include results and graphs here


# Citation

If you use this codebase, please cite our work.
