import json
import os
import torch
from tqdm import tqdm


from src.tools.tools import eval_neg_seq_len, eval_frac_0_samples
from .audio_attack_model_wrapper import AudioAttackModelWrapper

class AudioBaseAttacker():
    '''
        Base class for whitebox attack on Whisper Model in raw audio space
    '''
    def __init__(self, attack_args, model, device):
        self.attack_args = attack_args
        self.whisper_model = model
        self.device = device

        # model wrapper with audio attack segment prepending ability
        self.audio_attack_model = AudioAttackModelWrapper(self.whisper_model.tokenizer, attack_size=attack_args.attack_size, device=device).to(device)

    def eval_uni_attack(self, data, attack_model_dir=None, attack_epoch=-1, cache_dir=None, force_run=False):
        '''
            Generates transcriptions with audio attack segment (saves to cache)
            Computes the (negative average sequence length) = -1*mean(len(prediction))

            audio_attack_model is the directory with the saved audio_attack_model checkpoints with the attack audio values
            attack_epoch indicates the checkpoint of the learnt attack from training that should be used
                -1 indicates that no-attack should be evaluated
        '''
        # check for cache
        fpath = f'{cache_dir}/epoch-{attack_epoch}_predictions.json'
        if os.path.isfile(fpath) and not force_run:
            with open(fpath, 'r') as f:
                hyps = json.load(f)
            nsl, frac0 = eval_neg_seq_len(hyps), eval_frac_0_samples(hyps)
            return nsl, frac0
        
        # no cache
        if attack_epoch == -1:
            do_attack = False
        else:
            # load model with attack vector -- note if epoch=0, that is a rand prepend attack
            do_attack = True
            if attack_epoch > 0:
                self.audio_attack_model.load_state_dict(torch.load(f'{attack_model_dir}/epoch{attack_epoch}/model.th'))

        hyps = []
        for sample in tqdm(data):
            with torch.no_grad():
                hyp = self.audio_attack_model.transcribe(self.whisper_model, sample['audio'], do_attack=do_attack)
            hyps.append(hyp)
        nsl, frac0 = eval_neg_seq_len(hyps), eval_frac_0_samples(hyps)

        if cache_dir is not None:
            with open(fpath, 'w') as f:
                json.dump(hyps, f)

        return nsl, frac0