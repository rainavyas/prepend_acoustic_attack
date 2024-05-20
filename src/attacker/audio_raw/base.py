import json
import os
import torch
from tqdm import tqdm


from src.tools.tools import eval_neg_seq_len, eval_frac_0_samples, eval_wer, eval_average_fraction_of_languages
from .audio_attack_model_wrapper import AudioAttackModelWrapper

class AudioBaseAttacker():
    '''
        Base class for whitebox attack on Whisper Model in raw audio space
    '''
    def __init__(self, attack_args, model, device, attack_init='random'):
        self.attack_args = attack_args
        self.whisper_model = model
        self.device = device

        # model wrapper with audio attack segment prepending ability
        self.audio_attack_model = AudioAttackModelWrapper(self.whisper_model.tokenizer, attack_size=attack_args.attack_size, device=device, attack_init=attack_init).to(device)

    def _get_tgt_tkn_id(self):
        if self.attack_args.attack_token == 'eot':
            return self.whisper_model.tokenizer.eot
        elif self.attack_args.attack_token == 'transcribe':
            return self.whisper_model.tokenizer.transcribe

    def evaluate_metrics(self, hyps, refs, metrics, frac_lang_languages):
        results = {}
        if 'nsl' in metrics:
            results['Negative Sequence Length'] = eval_neg_seq_len(hyps)
        if 'frac0' in metrics:
            results['Fraction 0 length'] = eval_frac_0_samples(hyps)
        if 'wer' in metrics:
            results['WER'] = eval_wer(hyps, refs)
        if 'frac_lang' in metrics:
            results['Fraction of Languages'] = eval_average_fraction_of_languages(hyps, frac_lang_languages)
        return results

    def eval_uni_attack(self, data, attack_model_dir=None, attack_epoch=-1, cache_dir=None, force_run=False, metrics=['nsl', 'frac0'], frac_lang_languages=['en', 'fr']):
        '''
            Generates transcriptions with audio attack segment (saves to cache)
            Computes the metrics specified
                nsl : negative sequence length (average)
                frac0 : fraction of samples that are 0
                wer: Word Error Rate
                frac_lang: fraction of specified languages (average over hyps)

            audio_attack_model is the directory with the saved audio_attack_model checkpoints with the attack audio values
            attack_epoch indicates the checkpoint of the learnt attack from training that should be used
                -1 indicates that no-attack should be evaluated
        '''
        # check for cache
        fpath = f'{cache_dir}/epoch-{attack_epoch}_predictions.json'
        if os.path.isfile(fpath) and not force_run:
            with open(fpath, 'r') as f:
                hyps = json.load(f)
            refs = [d['ref'] for d in data]
            return self.evaluate_metrics(hyps, refs, metrics, frac_lang_languages)
        
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

        refs = [d['ref'] for d in data]
        out = self.evaluate_metrics(hyps, refs, metrics, frac_lang_languages)

        if cache_dir is not None:
            with open(fpath, 'w') as f:
                json.dump(hyps, f)

        return out