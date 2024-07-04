import json
import os
import torch
from tqdm import tqdm


from src.tools.tools import eval_neg_seq_len, eval_frac_0_samples, eval_wer, eval_average_fraction_of_languages, eval_bleu, eval_bleu_dist, eval_comet, eval_comet_dist, eval_english_probability, eval_english_probability_dist, eval_bleu_english_prob_recall, eval_comet_english_prob_recall
from .audio_attack_model_wrapper import AudioAttackModelWrapper
from .audio_attack_canary_model_wrapper import AudioAttackCanaryModelWrapper

class AudioBaseAttacker():
    '''
        Base class for whitebox attack on Whisper Model in raw audio space
    '''
    def __init__(self, attack_args, model, device, attack_init='random'):
        self.attack_args = attack_args
        self.whisper_model = model # may be canary model
        self.device = device

        # model wrapper with audio attack segment prepending ability
        if 'whisper' in model.model_name:
            self.audio_attack_model = AudioAttackModelWrapper(self.whisper_model.tokenizer, attack_size=attack_args.attack_size, device=device, attack_init=attack_init).to(device)
        elif 'canary' in model.model_name:
            self.audio_attack_model = AudioAttackCanaryModelWrapper(self.whisper_model.tokenizer, attack_size=attack_args.attack_size, device=device, attack_init=attack_init).to(device)


    def _get_tgt_tkn_id(self):
        if self.attack_args.attack_token == 'eot':
            try:
                eot_id =  self.whisper_model.tokenizer.eot
            except:
                # canary model
                eot_id = self.whisper_model.tokenizer.eos_id
            return eot_id
        elif self.attack_args.attack_token == 'transcribe':
            return self.whisper_model.tokenizer.transcribe

    def evaluate_metrics(self, hyps, refs_data, metrics, frac_lang_languages, attack=False):
        refs = [d['ref'] for d in refs_data]
        results = {}
        if 'nsl' in metrics:
            results['Negative Sequence Length'] = eval_neg_seq_len(hyps)
        if 'frac0' in metrics:
            results['Fraction 0 length'] = eval_frac_0_samples(hyps)
        if 'wer' in metrics:
            results['WER'] = eval_wer(hyps, refs)
        if 'frac_lang' in metrics:
            results['Fraction of Languages'] = eval_average_fraction_of_languages(hyps, frac_lang_languages)
        if 'bleu' in metrics:
            results['BLEU'] = eval_bleu(hyps, refs)
        if 'bleu_dist' in metrics:
            _ = eval_bleu_dist(hyps, refs, attack=attack)
            print('BLEU dist files generated in experiments/plots/')
        if 'comet' in metrics:
            srcs = [d['ref_src'] for d in refs_data]
            results['COMET'] = eval_comet(srcs, hyps, refs)
        if 'comet_dist' in metrics:
            srcs = [d['ref_src'] for d in refs_data]
            _ = eval_comet_dist(srcs, hyps, refs, attack=attack)
            print('COMET dist files generated in experiments/plots/')
        if 'en_prob' in metrics:
            results['Prob en'] = eval_english_probability(hyps)
        if 'en_prob_dist' in metrics:
            _ = eval_english_probability_dist(hyps, attack=attack)
            print('prob(en) dist files generated in experiments/plots/')
        if 'bleu_en_prob_recall' in metrics:
            _ = eval_bleu_english_prob_recall(hyps, refs, attack=attack)
            _ = eval_bleu_english_prob_recall(hyps, refs, attack=attack, rev_attack=True)
            print('bleu recall generated in experiments/plots/')
        if 'comet_en_prob_recall' in metrics:
            srcs = [d['ref_src'] for d in refs_data]
            _ = eval_comet_english_prob_recall(srcs, hyps, refs, attack=attack)
            _ = eval_comet_english_prob_recall(srcs, hyps, refs, attack=attack, rev_attack=True)
            print('comet recall generated in experiments/plots/')

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
            return self.evaluate_metrics(hyps, data, metrics, frac_lang_languages, attack=attack_epoch!=-1)
        
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
        out = self.evaluate_metrics(hyps, data, metrics, frac_lang_languages)

        if cache_dir is not None:
            with open(fpath, 'w') as f:
                json.dump(hyps, f)

        return out