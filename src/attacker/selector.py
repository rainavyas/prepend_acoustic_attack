from .mel.soft_prompt_attack import SoftPromptAttack
from .mel.base import MelBaseAttacker

def select_eval_attacker(attack_args, core_args, model, device=None):
    if attack_args.attack_method == 'mel':
        # Whitebox ASR Mel Softprompt attack
        return MelBaseAttacker(attack_args, model, device)


def select_train_attacker(attack_args, core_args, model, word_list=None, device=None):
    if attack_args.attack_method == 'mel':
        # Whitebox ASR Mel Softprompt attack
        return SoftPromptAttack(attack_args, model, device)
   