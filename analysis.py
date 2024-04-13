import torch
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
    load_audio
)
from math import log

from src.tools.tools import get_default_device
from src.tools.args import core_args, attack_args, analysis_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval, attack_base_path_creator_train
from src.models.load_model import load_model
from src.data.load_data import load_data
from src.attacker.selector import select_eval_attacker

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    analysis_args, _ = analysis_args()

    print(core_args)
    print(attack_args)
    print(analysis_args)

    if not attack_args.transfer:
        base_path = base_path_creator(core_args)
        attack_base_path = attack_base_path_creator_eval(attack_args, base_path)
    else:
        base_path = None
        attack_base_path = None


    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analysis.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # Load the model
    model = load_model(core_args, device=device)

    # load attacker for evaluation
    attacker = select_eval_attacker(attack_args, core_args, model, device=device)

    # extract the attack model attack mel vectors
    attack_model = attacker.audio_attack_model
    if not attack_args.transfer:
        attack_model_dir = f'{attack_base_path_creator_train(attack_args, base_path)}/prepend_attack_models'
    else:
        attack_model_dir = attack_args.attack_model_dir
    attack_model.load_state_dict(torch.load(f'{attack_model_dir}/epoch{attack_args.attack_epoch}/model.th'))

    audio = attack_model.audio_attack_segment.cpu().detach()

    if analysis_args.compare_with_audio:
        _, data = load_data(core_args)
        audio_sample = data[analysis_args.sample_id]['audio']
        audio_sample = load_audio(audio_sample)
        audio_sample = torch.from_numpy(audio_sample)
        audio = torch.cat((audio, audio_sample), dim=0)

    # map to mel space
    log_adv_mel = log_mel_spectrogram(audio, model.model.dims.n_mels)
    adv_mel = np.exp(log_adv_mel.numpy())

    # plot the heatmap
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(adv_mel, y_axis='linear', x_axis='time', hop_length=HOP_LENGTH,
                                sr=SAMPLE_RATE)
    fig.colorbar(img, ax=ax, format="%.2f dB")

    # save image
    if analysis_args.compare_with_audio:
        ax.set_xlim(right=3)
        if attack_args.transfer:
            save_path = 'experiments/transfer_spectrogram.png'
        else:
            save_path = f'{attack_base_path}/comparison_spectrogram.png'
    else:
        save_path = f'{attack_base_path}/spectrogram.png'
    fig.savefig(save_path, bbox_inches='tight')