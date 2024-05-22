from whisper.normalizers import EnglishTextNormalizer
# import editdistance
import jiwer
import torch
import random


from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def get_default_device(gpu_id=0):
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device(f'cuda:{gpu_id}')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def eval_wer(hyps, refs, get_details=False):
    # assuming the texts are already aligned and there is no ID in the texts
    # WER
    std = EnglishTextNormalizer()
    hyps = [std(hyp) for hyp in hyps]
    refs = [std(ref) for ref in refs]
    out = jiwer.process_words(refs, hyps)
    
    total_ref = sum(len(ref.split()) for ref in refs)  # total number of words in the reference
    
    if not get_details:
        return out.wer
    else:
        # return ins, del and sub rates
        ins_rate = out.insertions / total_ref
        del_rate = out.deletions / total_ref
        sub_rate = out.substitutions / total_ref
        return {'WER': out.wer, 'INS': ins_rate, 'DEL': del_rate, 'SUB': sub_rate, 'HIT': out.hits/total_ref}




def eval_neg_seq_len(hyps):
    '''
        Average sequence length (negative)
    '''
    nlens = 0
    for hyp in hyps:
        nlens += (len(hyp.split()))
    return (-1)*nlens/len(hyps)

def eval_frac_0_samples(hyps):
    '''
        Fraction of samples of 0 tokens
    '''
    no_len_count = 0
    for hyp in hyps:
        if len(hyp) == 0:
            no_len_count +=1
    return no_len_count/len(hyps)

def eval_average_fraction_of_languages(sentences, languages):
    """
    Calculates the average fraction of specified languages across a list of sentences.
    
    Args:
        sentences (list of str): A list of sentences to analyze.
        languages (list of str): A list of language codes to detect (e.g., 'en' for English, 'fr' for French).
    
    Returns:
        dict: A dictionary where keys are language codes and values are the average fractions of each language.
    """
    total_language_counts = {lang: 0 for lang in languages}
    total_words = 0
    
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        language_counts = {lang: 0 for lang in languages}
        
        for word in words:
            try:
                detected_language = detect(word)
                if detected_language in language_counts:
                    language_counts[detected_language] += 1
            except LangDetectException:
                # Skip words that cannot be detected
                continue
        
        for lang in languages:
            total_language_counts[lang] += language_counts[lang]
    
    if total_words == 0:
        return {lang: 0 for lang in languages}
    
    average_language_fractions = {lang: total_language_counts[lang] / total_words for lang in languages}
    return average_language_fractions

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
