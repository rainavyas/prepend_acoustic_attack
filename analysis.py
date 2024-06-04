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
from statistics import mean, stdev
from tqdm import tqdm

from src.tools.tools import get_default_device, eval_wer, eval_neg_seq_len
from src.tools.args import core_args, attack_args, analysis_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval, attack_base_path_creator_train
from src.tools.analysis_tools import saliency, frame_level_saliency, get_decoder_proj_mat, get_rel_pos, get_real_acoustic_token_ids
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

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analysis.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    if analysis_args.model_emb_close_exs:
        '''
            Print 10 closest words as per the projection matrix (same as embedding matrix)
            for the selected tokens
        '''

        def print_closest_words(rel_pos, tokenizer, words, top_n=10):
            for word in words:
                word_id = tokenizer.encode(word, disallowed_special=())[0]
                closest_ids = torch.argsort(-rel_pos[word_id])[:top_n+1]  # +1 to account for the word itself
                closest_words = [tokenizer.decode([i]) for i in closest_ids if i != word_id]
                print(f'The {top_n} closest words to "{word}": {closest_words[:top_n]}')

        # Get the device
        if core_args.force_cpu:
            device = torch.device('cpu')
        else:
            device = get_default_device(core_args.gpu_id)
        print(device)

        # Load the model
        whisper_model = load_model(core_args, device=device)
        model = whisper_model.model

        # Get projection matrix
        W = get_decoder_proj_mat(model)

        # Get relative position vector for each token wrt to real acoustic sounding tokens
        real_token_ids = get_real_acoustic_token_ids(whisper_model.tokenizer, vocab_size=W.size(0))
        rel_pos = get_rel_pos(W, real_token_ids)

        # Define the words of interest
        words_of_interest = ['zoo', 'boy', 'hi']

        # Add the EOT token to the list of words of interest
        eot_token = whisper_model.tokenizer.eot
        eot_word = whisper_model.tokenizer.decode([eot_token])
        words_of_interest.append(eot_word)

        # Print the 10 closest words for each word of interest
        print_closest_words(rel_pos, whisper_model.tokenizer, words_of_interest, top_n=10)
        

    if analysis_args.model_transfer_check:
        '''
            Determine if a muting adversarial attack can transfer between two models.

            Compute the relative position of the eot token relative to other tokens using the projection matrix in the final layer of the decoder.
        '''

    
        # Get the device
        if core_args.force_cpu:
            device = torch.device('cpu')
        else:
            device = get_default_device(core_args.gpu_id)
        print(device)

        # Load the two models (m and n)
        whisper_model = load_model(core_args)
        model_m = whisper_model.models[0]
        model_n = whisper_model.models[1]

        # get projection matrices
        W_m = get_decoder_proj_mat(model_m)
        W_n = get_decoder_proj_mat(model_n)

        # get relative position vector for each token wrt to real acoustic sounding tokens
        real_token_ids = get_real_acoustic_token_ids(whisper_model.tokenizer, vocab_size=W_m.size(0))
        rel_pos_m = get_rel_pos(W_m, real_token_ids)
        rel_pos_n = get_rel_pos(W_n, real_token_ids)

        # measure change (score) in rel_pos across the different models
        diff = torch.linalg.norm(rel_pos_m - rel_pos_n, dim=1)

        # get the score for the target eot token
        eot_id = whisper_model.tokenizer.eot
        eot_score = diff[eot_id].item()
        print('EOT score:', eot_score)

        # Get the avg score + 2*std for the real token ids
        real_token_scores = diff[real_token_ids]

        # Calculate mean and standard deviation
        mean_score = real_token_scores.mean().item()
        std_score = real_token_scores.std().item()

        print(f'Real tokens score:\t{mean_score} +- {std_score}')


    if analysis_args.saliency_plot:
        '''
            Plot frame-level saliency
        '''
        base_path = base_path_creator(core_args)
        attack_base_path = attack_base_path_creator_eval(attack_args, base_path)
        attack_hyp_file = f"{attack_base_path}/epoch-{attack_args.attack_epoch}_predictions.json"

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

        # get indices of (un)/successful attack hyps
        with open(attack_hyp_file, 'r') as f:
            attack_hyps = json.load(f)
        
        inds = [i for i in range(len(attack_hyps)) if len(attack_hyps[i])==0]
        success_data = [test_data[ind] for ind in inds]
        success_sample = success_data[10]

        inds = [i for i in range(len(attack_hyps)) if len(attack_hyps[i])!=0]
        unsuccess_data = [test_data[ind] for ind in inds]
        unsuccess_sample = unsuccess_data[1]

        # Load the model
        whisper_model = load_model(core_args, device=device)

        # Load the attack model wrapper
        attacker = select_eval_attacker(attack_args, core_args, whisper_model, device=device)
        audio_attack_model = attacker.audio_attack_model
        attack_model_dir = f'{attack_base_path_creator_train(attack_args, base_path)}/prepend_attack_models'
        audio_attack_model.load_state_dict(torch.load(f'{attack_model_dir}/epoch{attack_args.attack_epoch}/model.th'))

        # compute frame-level saliency
        success_saliencies = frame_level_saliency(success_sample['audio'], audio_attack_model, whisper_model, device)
        unsuccess_saliencies = frame_level_saliency(unsuccess_sample['audio'], audio_attack_model, whisper_model, device)

        # plot
        time_index = [i/SAMPLE_RATE for i in range(len(success_saliencies))]
        plt.plot(time_index, success_saliencies)
        plt.xlabel('Time')
        plt.ylabel('Saliency')
        plt.xlim((0,3))
        plt.vlines(x=0.64, colors='red', ymin=0, ymax=max(success_saliencies), linestyles='dashed')
        save_path = f'{attack_base_path}/success_saliency.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()

        time_index = [i/SAMPLE_RATE for i in range(len(unsuccess_saliencies))]
        plt.plot(time_index, unsuccess_saliencies)
        plt.xlabel('Time')
        plt.ylabel('Saliency')
        plt.xlim((0,3))
        plt.vlines(x=0.64, colors='red', ymin=0, ymax=max(unsuccess_saliencies), linestyles='dashed')
        save_path = f'{attack_base_path}/unsuccess_saliency.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()


    if analysis_args.saliency:
        '''
            Compute the saliency of the adversarial segment an non-adversarial segment (average over dataset)
            Split for successful (0 length prediction) and unsuccessful attacks
        '''
        if not attack_args.transfer:
            base_path = base_path_creator(core_args)
            attack_base_path = attack_base_path_creator_eval(attack_args, base_path)
            attack_hyp_file = f"{attack_base_path}/epoch-{attack_args.attack_epoch}_predictions.json"
        else:
            base_path = None
            attack_base_path = None
            attack_hyp_file = analysis_args.attack_path

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

        # get indices of (un)/successful attack hyps
        with open(attack_hyp_file, 'r') as f:
            attack_hyps = json.load(f)
        
        inds = [i for i in range(len(attack_hyps)) if len(attack_hyps[i])==0]
        success_data = [test_data[ind] for ind in inds]
        
        inds = [i for i in range(len(attack_hyps)) if len(attack_hyps[i])!=0]
        unsuccess_data = [test_data[ind] for ind in inds]


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

        # Compute saliencies - unsuccessful attacks
        print('UNSUCCESSFUL ATTACKS')
        adv_sals = []
        non_adv_sals = []
        for sample in tqdm(unsuccess_data):
            adv_grad, non_adv_grad = saliency(sample['audio'], audio_attack_model, whisper_model, device)
            adv_sals.append(adv_grad)
            non_adv_sals.append(non_adv_grad)
        
        print(f"Num Samples\t{len(unsuccess_data)}")
        print(f"Adv Saliency:\t{mean(adv_sals)}\t+-\t{stdev(adv_sals)}")
        print(f"Non-adv Saliency:\t{mean(non_adv_sals)}\t+-\t{stdev(non_adv_sals)}")
        print("-----------------------------------------------------")
        print()

        # Compute saliencies - successful attacks
        print('SUCCESSFUL ATTACKS')
        adv_sals = []
        non_adv_sals = []
        for sample in tqdm(success_data):
            adv_grad, non_adv_grad = saliency(sample['audio'], audio_attack_model, whisper_model, device)
            adv_sals.append(adv_grad)
            non_adv_sals.append(non_adv_grad)
        
        print(f"Num Samples\t{len(success_data)}")
        print(f"Adv Saliency:\t{mean(adv_sals)}\t+-\t{stdev(adv_sals)}")
        print(f"Non-adv Saliency:\t{mean(non_adv_sals)}\t+-\t{stdev(non_adv_sals)}")
        print("-----------------------------------------------------")
        print()


    if analysis_args.wer_no_0:
        '''
            Get the WER of non zero length samples for attacked samples (unsuccessful), 
            and also get equivalent WER for the same samples when not attacked.

            Repeat for successful samples
        '''
        # load data
        train_data, test_data = load_data(core_args)
        if attack_args.eval_train:
            test_data = train_data

        with open(analysis_args.no_attack_path, 'r') as f:
            no_attack_hyps = json.load(f)

        with open(analysis_args.attack_path, 'r') as f:
            attack_hyps = json.load(f)
        
        # get indices of unsuccessful attack hyps
        inds = [i for i in range(len(attack_hyps)) if len(attack_hyps[i])!=0]

        ahyps = [attack_hyps[ind] for ind in inds]
        no_ahyps = [no_attack_hyps[ind] for ind in inds]
        refs = [test_data[ind]['ref'] for ind in inds]

        # Samples
        print('UNSUCCESSFUL SAMPLES')
        print('------------------------------------')
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


        # repeat for successful samples

        # get indices of unsuccessful attack hyps
        inds = [i for i in range(len(attack_hyps)) if len(attack_hyps[i])==0]

        ahyps = [attack_hyps[ind] for ind in inds]
        no_ahyps = [no_attack_hyps[ind] for ind in inds]
        refs = [test_data[ind]['ref'] for ind in inds]

        # Samples
        print()
        print('SUCCESSFUL SAMPLES')
        print('------------------------------------')
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