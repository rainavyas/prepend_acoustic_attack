'''
 General small processing activities - extract and save audio segment
'''
import argparse
import torch
import os
import sys
import numpy as np

from src.attacker.audio_raw.audio_attack_model_wrapper import AudioAttackModelWrapper


def get_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--attack_model_path', type=str, default='', help='Full path to attack model')
    commandLineParser.add_argument('--save_path', type=str, default='', help='Full path for where to save numpy array')

    return commandLineParser.parse_known_args()

if __name__ == "__main__":

    args, _ = get_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/process.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # extract the audio attack vectors from pytorch and save as numpy array

    attack_model = AudioAttackModelWrapper(None, attack_size=10240)
    attack_model.load_state_dict(torch.load(f'{args.attack_model_path}'))
    audio = attack_model.audio_attack_segment.cpu().detach().numpy()
    np.save(args.save_path, audio)