from .mel.soft_prompt_attack import SoftPromptAttack
from .mel.base import MelBaseAttacker
from .audio_raw.base import AudioBaseAttacker
from .audio_raw.learn_attack import AudioAttack

def select_eval_attacker(attack_args, core_args, model, device=None):
    if attack_args.attack_method == 'mel':
        return MelBaseAttacker(attack_args, model, device)
    elif attack_args.attack_method == 'audio-raw':
        return AudioBaseAttacker(attack_args, model, device)


def select_train_attacker(attack_args, core_args, model, word_list=None, device=None):
    if attack_args.attack_method == 'mel':
        return SoftPromptAttack(attack_args, model, device)
    elif attack_args.attack_method == 'audio-raw':
        return AudioAttack(attack_args, model, device, lr=attack_args.lr)
   