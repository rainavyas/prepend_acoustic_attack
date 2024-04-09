import torch
from tqdm import tqdm
import json
import os

from whisper.tokenizer import get_tokenizer
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from src.tools.tools import eval_neg_seq_len
from .softprompt_model_wrapper import SoftPromptModelWrapper


class MelBaseAttacker():
    '''
        Base class for whitebox attack on Whisper Model in mel-vector space
    '''
    def __init__(self, attack_args, model, device):
        self.attack_args = attack_args
        self.whisper_model = model # assume it is a whisper model
        self.tokenizer = get_tokenizer(self.whisper_model.model.is_multilingual, num_languages=self.whisper_model.model.num_languages, task=self.whisper_model.task)
        self.device = device
        # model wrapper with softprompting ability in mel-vector space
        self.softprompt_model = SoftPromptModelWrapper(self.tokenizer, device=device).to(device)

    def audio_to_mel(self, audio):
        '''
            Get sequence of mel-vectors
        '''
        # 30s of silence added to audio and then calculate mel vectors
        mel = log_mel_spectrogram(audio, self.whisper_model.model.dims.n_mels, padding=N_SAMPLES)

        # truncate such that the total audio is of length N_FRAMES
        mel = pad_or_trim(mel, N_FRAMES)
        return mel
    
    def eval_uni_attack(self, data, softprompt_model_dir=None, attack_epoch=-1, k_scale=1, cache_dir=None, force_run=False):
        '''
            Generates transcriptions with softprompt_model learnt softprompts (saves to cache)
            Computes the (negative average sequence length) = -1*mean(len(prediction))

            model_dir is the directory with the saved softprompt_model checkpoints with the attack mel-vectors
            attack_epoch indicates the checkpoint of the learnt mel vectors from training that should be used
                -1 indicates that no-attack should be evaluated
        '''
        # check for cache
        if k_scale > 1:
            fpath = f'{cache_dir}/epoch-{attack_epoch}_k{k_scale}_predictions.json'
        else:
            fpath = f'{cache_dir}/epoch-{attack_epoch}_predictions.json'
        if os.path.isfile(fpath) and not force_run:
            with open(fpath, 'r') as f:
                hyps = json.load(f)
            
            nsl = eval_neg_seq_len(hyps)
            return nsl
        
        # no cache
        if attack_epoch == -1:
            do_mel_attack = False
        else:
            # load model with attack_mel vectors -- note if epoch=0, that is a rand mel vector attack
            do_mel_attack = True
            if attack_epoch > 0:
                self.softprompt_model.load_state_dict(torch.load(f'{softprompt_model_dir}/epoch{attack_epoch}/model.th'))

        hyps = []
        for sample in tqdm(data):
            with torch.no_grad():
                hyp = self.softprompt_model.transcribe(self.whisper_model, sample['audio'], do_mel_attack=do_mel_attack, k_scale=k_scale)
            hyps.append(hyp)
        nsl = eval_neg_seq_len(hyps)

        if cache_dir is not None:
            with open(fpath, 'w') as f:
                json.dump(hyps, f)

        return nsl