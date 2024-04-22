from whisper.normalizers import EnglishTextNormalizer
# import editdistance
import jiwer
import torch
import random

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
    if not get_details:
        return out.wer
    else:
        # return ins, del and sub rates
        total = out.insertions + out.deletions + out.substitutions + out.hits
        return {'WER':out.wer, 'INS':out.insertions/total, 'DEL':out.deletions/total, 'SUB':out.substitutions/total, 'HIT':out.hits/total}


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
