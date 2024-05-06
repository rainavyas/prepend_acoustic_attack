# Overview

This is the offical codebase for the Paper _Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models_

## Abstract
Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as _<endoftext>_, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's _<endoftext>_ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively _muting_ the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to _muting_ adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

# Try it out

We have uploaded all the pre-learnt universal acoustic adversarial attack segments in   `./audio_attacks/`. Open `demo.ipynb` and try evaluating them for yourself. Observe how these attacks can successfully mute Whisper models for unseen speech signals.

# Quick Start (Running the Code)


The following subsections give example commands to run the training, evaluations and analysis necessary to reproduce the results in our paper.

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



## Package Installation

This code has been tested on python>=3.9.

Fork the repository and then git clone

`git clone https://github.com/<username>/prepend_acoustic_attack`


Install all necessary packages by creating a conda environment from the existing `environment.yml` file.

```
conda env create -f environment.yml
conda activate venv_gector
```

## Standard Arguments for Attack Configuration

You can see all the arguments used in the different scripts in `src/tools/args.py`.

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

`python train_attack.py --model_name whisper-base-multi --data_name librispeech --attack_method audio-raw --max_epochs 40 --clip_val 0.02 --attack_size 10240 --save_freq 10`

## Evaluating a universal prepend acoustic attack

`eval_attack.py` is used to evaluate the efficacy of the attacks. Running the evaluation script evaluates the _no attack_ setting (the test audio samples are not modified) and the _attack_ setting (the test audio samples have the universal acoustic attack segment prepended in the raw audio space). For each setting, two metrics are reported:

1. **NSL** (Negative Sequence Length) - The (negative) average sequence length in words of the model predictions. The adversary aims to maximize this (as close to 0 as possible).
2. **frac 0** - The fraction of test samples for which the predicted sequence length was of length 0. This represents the fraction of _fully successful_ adversarial attacks, as the universal attack successfully _mutes_ the Whisper model at test time on these unseen test samples.

During evaluation the following extra arguments may be of use:

- `attack_epoch` : Each universal attack is trained for multiple epochs (until `max_epochs`). This argument allows you to select which trained version of the attack to evaluate. Note that you would have to set an appropriate `save_freq` for the trained universal attack segment to be saved at the selected `attack_epoch`
  
- `not-none` : Simply pass this argument if you do not want the evaluation script to evaluate the _no attack_ setting.

An example command for evaluation is given below:

`python eval_attack.py --model_name whisper-medium-multi --data_name librispeech --attack_method audio-raw --clip_val 0.02 --attack_size 10240 --attack_epoch 160 --not_none`


### Transfer attack evaluation

Beyond just evaluating the learnt universal adversarial attacks in the _matched_ setting, where the same dataset (attack is learnt on the validation split and evaluated on the test split) and same model are used, we can also evaluate how well the attack _transfers_ to different datasets and even tasks.

To evaluate the transferability the following further arguments are required:

- `transfer` : Simply pass this flag to indicate it is a transferability experiment

- `attack_model_dir` : This specifies the path to the model directory with the saved model wrapper containing the learnt universal attack segment. During training these directories and paths are automatically created. Refer to the example below to see the typical structure of these paths.

The below example looks at the transferability of an attack learnt on _librispeech_ to the _tedlium_ dataset.

`python eval_attack.py --model_name whisper-medium --data_name tedlium --attack_method audio-raw --attack_epoch 160 --attack_size 10240 --transfer --attack_model_dir experiments/librispeech/whisper-medium/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/ --not_none`


The next examples looks at the transferability of an attack learnt on _librispeech_ for the _transcribe_ task, to the _fleurs_ French dataset for the _translate_ task.

`python eval_attack.py --model_name whisper-tiny-multi --data_name fleurs --attack_size 10240 --language fr --task translate --attack_method audio-raw --attack_epoch 40 --transfer --attack_model_dir experiments/librispeech/whisper-tiny-multi/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/ --not_none`


## Analysis

Various forms of analysis are conducted in the paper. Here we give the commands used to generate the numbers given in the paper.


# Data

Describe here how certain datasets need to pre-downloaded and stored in particular directory. Also specify which part of the code needs to be updated to point to this directory.

# Results

include results and graphs here


# Citation

If you use this codebase, please cite our work.
