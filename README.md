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

## Standard Arguments for Attack Configuration

The following arguments specify the attack configuration:

- `model_name` : The specific Whisper model to learn the universal attack on.
- `attack_method`: What form of acoustic attack to learn. For this paper, we always use `audio-raw`.
- `clip_val` : The maximum amplitude (for imperceptibility) of the attack audio segment. Set to `0.02` in the paper.
- `attack_size` : The number of audio frames in the adversarial audio segment. Standard setting is `10,240`, which is equivalent to 0.64 seconds of adversarial audio, for audio sampled at 16kHz.
- `data_name` : The dataset on which the universal attack is to be trained / evaluated. Note that training is on the validation split of the dataset.
- `task` : This can either be `transcribe` or `translate`. This specifies the task that the Whisper model is required to do. Note that `translate` is only possible for the multi-lingual models.
- `language`: The source audio language. By default is `en`.

## Learning a universal prepend acoustic attack

`train_attack.py` can be used to learn a universal acoustic attack on any of the Whisper models. The following extra arguments may be of use:

- `max_epochs` : Maximum number of epochs to run the gradient-descent based training to learn the universal attack. In the paper we have the following configurations: tiny (40), base (40), small (120) and medium (160).
- `bs` : The batch size for learning the attack
- `save_freq` : The frequency of saving the learnt attack audio segment during the learning of the attack.

An example command for learning an attack is given below.

`python train_attack.py --model_name whisper-base-multi --attack_method audio-raw --max_epochs 40 --clip_val 0.02 --attack_size 10240 --save_freq 10`

## Evaluating a universal prepend acoustic attack

### Transfer attack evaluation

## Analysis


# Data

Describe here how certain datasets need to pre-downloaded and stored in particular directory. Also specify which part of the code needs to be updated to point to this directory.

# Results include results and graphs here


# Citation

If you use this codebase, please cite our work.
