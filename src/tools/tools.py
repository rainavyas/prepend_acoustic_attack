from whisper.normalizers import EnglishTextNormalizer
# import editdistance
import jiwer
import torch
import random
from tqdm import tqdm
import seaborn as sns

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from comet import download_model, load_from_checkpoint


from langdetect import detect, DetectorFactory, detect_langs
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

def get_english_probability(text):
    """
    Returns the probability of the given text being in English.
    
    :param text: The text to analyze
    :return: Probability of the text being in English
    """
    try:
        # Detect the languages and their probabilities
        languages = detect_langs(text)
        
        # Find the English probability
        for lang in languages:
            if lang.lang == 'en':
                return lang.prob
        
        # If English is not found, return 0
        return 0.0
    except Exception as e:
        # Handle cases where language detection fails
        print(f"Error detecting language: {e}")
        return 0.0

def eval_english_probability(hyps):
    """
    Returns the average probability of English for a list of sentences.
    
    :param hyps: List of sentences
    :return: Average probability of the sentences being in English
    """
    if not hyps:
        return 0.0
    
    total_prob = 0.0
    for sentence in hyps:
        total_prob += get_english_probability(sentence)
    
    return total_prob / len(hyps)

def eval_english_probability_dist(hyps, attack=False):
    """
    Computes the probability of each sentence being in English for a list of sentences,
    plots the distribution with KDE, and saves it to a file called 'experiments/plots/english_prob_dist.png'.

    Args:
        hyps (list of str): List of hypothesis strings.

    Returns:
        float: Overall average probability of the sentences being in English.
    """
    if not hyps:
        return 0.0

    # Compute the probabilities
    probabilities = [get_english_probability(sentence) for sentence in hyps]

    # Plot the distribution of probabilities with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(probabilities, bins=20, color='blue', edgecolor='black', kde=True, stat='density', alpha=0.3)
    sns.kdeplot(probabilities, color='blue', linewidth=2)
    plt.xlabel('Probability of being in English')
    plt.ylabel('Density')
    
    # Save the plot to a file
    fpath = 'experiments/plots/english_prob_dist_no_attack.png'
    if attack:
        fpath = 'experiments/plots/english_prob_dist_attack.png'
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()

    # Return the average probability
    return sum(probabilities) / len(probabilities)

def eval_bleu(hyps, refs):
    """
    Computes the BLEU score for a list of hypotheses and references.

    Args:
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        float: Overall BLEU score.
    """
    # Tokenize the sentences
    hyps = [hyp.split() for hyp in hyps]
    refs = [[ref.split()] for ref in refs]  # Note: refs need to be a list of lists of lists for sentence_bleu

    # Compute the overall BLEU score
    total_bleu = 0
    for hyp, ref in zip(hyps, refs):
        total_bleu += sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
    return total_bleu / len(hyps)

def eval_bleu_dist(hyps, refs, attack=False):
    """
    Computes the sentence-level BLEU scores for a list of hypotheses and references,
    plots the distribution, and saves it to a file called 'experiments/plots/bleu_dist.png'.

    Args:
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        list of float: List of sentence-level BLEU scores.
    """
    # Tokenize the sentences
    hyps = [hyp.split() for hyp in hyps]
    refs = [[ref.split()] for ref in refs]  # Note: refs need to be a list of lists of lists for sentence_bleu

    # Compute sentence-level BLEU scores
    bleu_scores = []
    for hyp, ref in zip(hyps, refs):
        score = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(score)

    # Plot the distribution of BLEU scores
    plt.figure(figsize=(10, 6))
    sns.histplot(bleu_scores, bins=20, color='blue', edgecolor='black', kde=True, stat='density', alpha=0.3)
    sns.kdeplot(bleu_scores, color='blue', linewidth=2)
    plt.xlabel('COMET score')
    plt.ylabel('Density')
    
    # Save the plot to a file
    fpath = 'experiments/plots/bleu_dist_no_attack.png'
    if attack:
        fpath = 'experiments/plots/bleu_dist_attack.png'
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()

    return sum(bleu_scores) / len(bleu_scores)


def eval_comet(srcs, hyps, refs):
    """
    Computes the overall COMET score for a list of sources, hypotheses, and references.

    Args:
        srcs (list of str): List of source strings.
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        float: Overall COMET score.
    """
    # Load the pre-trained COMET model for evaluation
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)

    # Prepare input data
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(srcs, hyps, refs)]

    # Compute COMET

    scores = model.predict(data, batch_size=8)['scores']
    scores = [s if s>0 else 0 for s in scores]
    return sum(scores)/len(scores) # average COMET score


def eval_comet_dist(srcs, hyps, refs, attack=False):
    """
    Computes the sentence-level COMET scores for a list of sources, hypotheses, and references,
    plots the distribution, and saves it to a file called 'experiments/plots/comet_dist.png'.

    Args:
        srcs (list of str): List of source strings.
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        float: Overall average COMET score.
    """
    # Load the pre-trained COMET model for evaluation
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)

    # Prepare input data
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(srcs, hyps, refs)]

    # Compute COMET scores
    comet_scores = model.predict(data, batch_size=8)['scores']
    comet_scores = [s if s>0 else 0 for s in comet_scores]

    # Plot the distribution of COMET scores
    plt.figure(figsize=(10, 6))
    sns.histplot(comet_scores, bins=20, color='blue', edgecolor='black', kde=True, stat='density', alpha=0.3)
    sns.kdeplot(comet_scores, color='blue', linewidth=2)
    plt.xlabel('COMET score')
    plt.ylabel('Density')
    
    # Save the plot to a file
    fpath = 'experiments/plots/comet_dist_no_attack.png'
    if attack:
        fpath = 'experiments/plots/comet_dist_attack.png'
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()

    # Return the average COMET score
    return sum(comet_scores) / len(comet_scores)



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


def eval_bleu_english_prob_recall(hyps, refs, attack=False, rev_attack=False):
    """
    Computes the BLEU scores and English probabilities for given samples, ranks the samples 
    by their probability of being English, and computes the average BLEU score up to each rank.
    Plots the 'BLEU score of samples classified as English (successfully attacked)' against 
    the '% of samples classified as English (successfully attacked)'.

    If rev_attack = True, calculates BLEU score for samples unsuccessfully attacked
    Args:
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.
        attack (bool): Flag indicating whether the attack mode is enabled. If False, the function 
                       will not run and return None.

    Returns:
        None
    """
    if not attack:
        return None

    # Compute BLEU scores
    hyps_tok = [hyp.split() for hyp in hyps]
    refs_tok = [[ref.split()] for ref in refs]  # refs need to be a list of lists of lists for sentence_bleu
    bleu_scores = [
        sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
        for hyp, ref in zip(hyps_tok, refs_tok)
    ]

    # Compute English probabilities
    english_probs = [get_english_probability(hyp) for hyp in hyps]

    # Combine BLEU scores and English probabilities
    combined = list(zip(bleu_scores, english_probs))
    
    # Rank samples by probability of English (highest to lowest) -- if rev_attack then in reverse
    combined.sort(key=lambda x: x[1], reverse=not rev_attack)

    # Compute average BLEU score up to each rank
    avg_bleu_scores = []
    percent_en = []
    cumulative_sum = 0.0
    for i, (bleu_score, en_prob) in enumerate(combined, start=1):
        cumulative_sum += bleu_score
        avg_bleu_scores.append(cumulative_sum / i)
        percent_en.append((i / len(combined)) * 100)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(percent_en, avg_bleu_scores, marker='o', linestyle='-', color='b')
    if rev_attack:
        plt.xlabel('Samples Classed as NOT En (%)')
        plt.ylabel('Average BLEU Score of Samples Classed as NOT En')
        save_path = 'experiments/plots/bleu_vs_not_en_probability.png'
    else:
        plt.xlabel('Samples Classed as En (%)')
        plt.ylabel('Average BLEU Score of Samples Classed as En')
        save_path = 'experiments/plots/bleu_vs_en_probability.png'
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return None


def eval_comet_english_prob_recall(srcs, hyps, refs, attack=False, rev_attack=False):
    """
    Computes the COMET scores and English probabilities for given samples, ranks the samples 
    by their probability of being English, and computes the average COMET score up to each rank.

    Args:
        srcs (list of str): List of source strings.
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.
        attack (bool): Flag indicating whether the attack mode is enabled. If False, the function 
                       will not run and return None.
        if rev_attack: COMET score of unsuccessfully attacked samples

    Plots:
        dict: A dictionary with two keys:
              - 'comet_scores': List of average COMET scores for the top-k samples by English probability.
              - 'percent_en': List of percentages of samples classified as English up to each k.
    """
    if not attack:
        return None

    # Load the pre-trained COMET model for evaluation
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)

    # Prepare input data
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(srcs, hyps, refs)]

    # Compute COMET scores
    comet_scores = model.predict(data, batch_size=8)['scores']
    comet_scores = [s if s > 0 else 0 for s in comet_scores]

    # Compute English probabilities
    english_probs = [get_english_probability(hyp) for hyp in hyps]

    # Combine COMET scores and English probabilities
    combined = list(zip(comet_scores, english_probs))
    
    # Rank samples by probability of English (highest to lowest)
    combined.sort(key=lambda x: x[1], reverse=not rev_attack)

    # Compute average COMET score up to each rank
    avg_comet_scores = []
    percent_en = []
    cumulative_sum = 0.0
    for i, (comet_score, en_prob) in enumerate(combined, start=1):
        cumulative_sum += comet_score
        avg_comet_scores.append(cumulative_sum / i)
        percent_en.append((i / len(combined)) * 100)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(percent_en, avg_comet_scores, marker='o', linestyle='-', color='b')
    if rev_attack:
        plt.xlabel('Samples Classed as NOT En (%)')
        plt.ylabel('Average COMET Score of Samples Classed as NOT En')
        save_path = 'experiments/plots/comet_vs_not_en_probability.png'
    else:
        plt.xlabel('Samples Classed as En (%)')
        plt.ylabel('Average COMET Score of Samples Classed as En')
        save_path = 'experiments/plots/comet_vs_en_probability.png'
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return None


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
