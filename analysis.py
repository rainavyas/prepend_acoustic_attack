import torch
import json
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
from statistics import mean, stdev
from tqdm import tqdm

from src.tools.tools import get_default_device, eval_wer, eval_neg_seq_len
from src.tools.args import core_args, attack_args, analysis_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval, attack_base_path_creator_train
from src.tools.analysis_tools import saliency
from src.models.load_model import load_model
from src.data.load_data import load_data
from src.attacker.selector import select_eval_attacker
from src.attacker.selector import select_eval_attacker

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    analysis_args, _ = analysis_args()

    print(core_args)
    print(attack_args)
    print(analysis_args)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analysis.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    

    if analysis_args.saliency:
        '''
            Compute the saliency of the adversarial segment an non-adversarial segment (average over dataset)
        '''
        base_path = base_path_creator(core_args)

        # Get the device
        if core_args.force_cpu:
            device = torch.device('cpu')
        else:
            device = get_default_device(core_args.gpu_id)
        print(device)

        # load data
        train_data, test_data = load_data(core_args)
        if attack_args.eval_train:
            test_data = train_data

        # Load the model
        whisper_model = load_model(core_args, device=device)

        # Load the attack model wrapper
        attacker = select_eval_attacker(attack_args, core_args, whisper_model, device=device)
        audio_attack_model = attacker.audio_attack_model
        if not attack_args.transfer:
            attack_model_dir = f'{attack_base_path_creator_train(attack_args, base_path)}/prepend_attack_models'
        else:
            attack_model_dir = attack_args.attack_model_dir
        audio_attack_model.load_state_dict(torch.load(f'{attack_model_dir}/epoch{attack_args.attack_epoch}/model.th'))

        # Compute saliencies
        adv_sals = []
        non_adv_sals = []
        for sample in tqdm(test_data):
            adv_grad, non_adv_grad = saliency(sample['audio'], audio_attack_model, whisper_model, device)
            adv_sals.append(adv_grad)
            non_adv_sals.append(non_adv_grad)
        
        print(f"Adv Saliency:\t{mean(adv_sals)}\t+-\t{stdev(adv_sals)}")
        print(f"Non-adv Saliency:\t{mean(non_adv_sals)}\t+-\t{stdev(non_adv_sals)}")



    if analysis_args.wer_no_0:
        '''
            Get the WER of non zero length samples for attacked samples, 
            and also get equivalent WER for the same samples when not attacked.
        '''
        # load data
        train_data, test_data = load_data(core_args)
        if attack_args.eval_train:
            test_data = train_data

        with open(analysis_args.no_attack_path, 'r') as f:
            no_attack_hyps = json.load(f)

        with open(analysis_args.attack_path, 'r') as f:
            attack_hyps = json.load(f)
        
        # get indices of non zero attack hyps
        inds = [i for i in range(len(attack_hyps)) if len(attack_hyps[i])!=0]

        ahyps = [attack_hyps[ind] for ind in inds]
        no_ahyps = [no_attack_hyps[ind] for ind in inds]
        refs = [test_data[ind]['ref'] for ind in inds]


        # Samples
        print(f'Num Samples:\t{len(refs)}')
        print()

        # ASL
        no_a_asl = -1*eval_neg_seq_len(no_ahyps)
        a_asl = -1*eval_neg_seq_len(ahyps)
        print(f'No Attack ASL:\t{no_a_asl}')
        print(f'Attack ASL:\t{a_asl}')
        print()

        # no attack vs ref
        out = eval_wer(no_ahyps, refs, get_details=True)
        print('No attack vs ref')
        print(f'WER:\t{out["WER"]*100}')
        print(f'INS:\t{out["INS"]*100}')
        print(f'DEL:\t{out["DEL"]*100}')
        print(f'SUB:\t{out["SUB"]*100}')
        print()

        # attack vs ref
        out = eval_wer(ahyps, refs, get_details=True)
        print('Attack vs ref')
        print(f'WER:\t{out["WER"]*100}')
        print(f'INS:\t{out["INS"]*100}')
        print(f'DEL:\t{out["DEL"]*100}')
        print(f'SUB:\t{out["SUB"]*100}')
        print()

        # no attack vs attack
        out = eval_wer(ahyps, no_ahyps, get_details=True)
        print('Attack vs No Attack')
        print(f'WER:\t{out["WER"]*100}')
        print(f'INS:\t{out["INS"]*100}')
        print(f'DEL:\t{out["DEL"]*100}')
        print(f'SUB:\t{out["SUB"]*100}')
        print()


    if analysis_args.spectrogram:

        if not attack_args.transfer:
            base_path = base_path_creator(core_args)
            attack_base_path = attack_base_path_creator_eval(attack_args, base_path)
        else:
            base_path = None
            attack_base_path = None


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