"""
    Evaluate attack
"""

import sys
import os
import torch
import numpy as np

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import (
    base_path_creator,
    attack_base_path_creator_eval,
    attack_base_path_creator_train,
)
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.attacker.selector import select_eval_attacker

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    print(core_args)
    print(attack_args)

    set_seeds(core_args.seed)
    if not attack_args.transfer:
        base_path = base_path_creator(core_args)
        attack_base_path = attack_base_path_creator_eval(attack_args, base_path)
    else:
        base_path = None
        attack_base_path = None

    # Save the command run
    if not os.path.isdir("CMDs"):
        os.mkdir("CMDs")
    with open("CMDs/eval_attack.cmd", "a") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Get the device
    if core_args.force_cpu:
        device = torch.device("cpu")
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # Load the data
    train_data, test_data = load_data(core_args)
    if attack_args.eval_train:
        test_data = train_data

    # Load the model
    model = load_model(core_args, device=device)

    # load attacker for evaluation
    attacker = select_eval_attacker(attack_args, core_args, model, device=device)

    # evaluate
    if not attack_args.transfer:
        attack_model_dir = f"{attack_base_path_creator_train(attack_args, base_path)}/prepend_attack_models"
    else:
        attack_model_dir = attack_args.attack_model_dir

    # only_wer = attack_args.only_wer
    # if attack_args.attack_token == 'transcribe':
    #     only_wer = True

    # 1) No attack
    if not attack_args.not_none:
        print("No attack")
        out = attacker.eval_uni_attack(
            test_data,
            attack_model_dir=attack_model_dir,
            attack_epoch=-1,
            cache_dir=attack_base_path,
            force_run=attack_args.force_run,
            metrics=attack_args.eval_metrics,
            frac_lang_languages=attack_args.frac_lang_langs,
        )
        print(out)
        print()

    # 2) Attack
    print("Attack")
    out = attacker.eval_uni_attack(
        test_data,
        attack_model_dir=attack_model_dir,
        attack_epoch=attack_args.attack_epoch,
        cache_dir=attack_base_path,
        force_run=attack_args.force_run,
        metrics=attack_args.eval_metrics,
        frac_lang_languages=attack_args.frac_lang_langs,
    )
    print(out)
    print()
