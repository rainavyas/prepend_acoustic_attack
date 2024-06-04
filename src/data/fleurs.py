import os
import json
from datasets import load_dataset
from src.models.whisper import WhisperModel
from src.tools.tools import get_default_device
from tqdm import tqdm

LANG_MAPPER = {
    'fr': 'fr_fr',
    'de': 'de_de',
    'ru': 'ru_ru',
    'ko': 'ko_kr',
    'en': 'en_us'
}

CACHE_DIR = "data/fleurs"

def _fleurs(lang='fr', use_pred_for_ref=False, model_name=None, gpu_id=0):
    """
    Loads the FLEURS dataset for a given language or language pair.

    Args:
        lang (str): The language code or language pair code in the format 'src_tgt'.
        use_pred_for_ref (bool): Whether to use Whisper model predictions for the 'ref' content.
        model_name (str): The name of the Whisper model to use for predictions.
        gpu_id (int): The GPU ID to use for the device.

    Returns:
        tuple: Parallel training and test samples if a language pair is specified,
               otherwise training and test samples for the specified language.
    """
    device = get_default_device(gpu_id)
    if '_' in lang:
        return _fleurs_parallel(lang, use_pred_for_ref, model_name, device)
    else:
        train_data = load_dataset("google/fleurs", f"{LANG_MAPPER[lang]}", split="train")
        test_data = load_dataset("google/fleurs", f"{LANG_MAPPER[lang]}", split="test")
        return _prep_samples(train_data, use_pred_for_ref, model_name, lang, "transcribe", device), _prep_samples(test_data, use_pred_for_ref, model_name, lang, "transcribe", device)

def _prep_samples(data, use_pred_for_ref, model_name, lang, task, device):
    """
    Prepares samples with audio paths and transcriptions.

    Args:
        data (Dataset): The dataset to prepare samples from.
        use_pred_for_ref (bool): Whether to use Whisper model predictions for the 'ref' content.
        model_name (str): The name of the Whisper model to use for predictions.
        lang (str): The language code for the source audio language.
        task (str): The task for the Whisper model ('transcribe' or 'translate').
        device (torch.device): The device to run the Whisper model on.

    Returns:
        list: List of dictionaries containing 'ref' and 'audio' keys.
    """
    cache_dir = os.path.join(CACHE_DIR, lang, model_name or "default", task)
    os.makedirs(cache_dir, exist_ok=True)
    
    samples = []
    whisper_model = None
    if use_pred_for_ref:
        whisper_model = WhisperModel(model_name=model_name, task=task, language=lang, device=device)
        print(f"Generating predictions for the dataset ({task})...")

    for sample in tqdm(list(data), desc="Processing samples"):
        audio_path = '/'.join(sample['path'].split('/')[:-1]) + '/' + sample['audio']['path']
        cache_file = os.path.join(cache_dir, f"{sample['id']}.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                ref = json.load(f)['ref']
        else:
            ref = sample['transcription']
            if use_pred_for_ref and whisper_model:
                ref = whisper_model.predict(audio=audio_path)
                with open(cache_file, 'w') as f:
                    json.dump({'ref': ref}, f)
        
        samples.append(
            {
                'id': sample['id'],
                'ref': ref,
                'audio': audio_path
            }
        )
    return samples

def _fleurs_parallel(lang_pair='fr_en', use_pred_for_ref=False, model_name=None, device=None):
    """
    Loads parallel sentences between the specified source and target languages from the FLEURS dataset.

    This function prepares the dataset for a spoken language translation task, where each sample contains:
    - 'audio': Audio of the source language.
    - 'ref': Transcription of the target language.
    - 'audio_tgt': Audio of the target language.
    - 'ref_src': Transcription of the source language.

    Args:
        lang_pair (str): The language pair code in the format 'src_tgt'.
        use_pred_for_ref (bool): Whether to use Whisper model predictions for the 'ref' content.
        model_name (str): The name of the Whisper model to use for predictions.
        device (torch.device): The device to run the Whisper model on.

    Returns:
        tuple: Parallel training and test samples.
    """
    src_lang, tgt_lang = lang_pair.split('_')
    
    # Load datasets for source and target languages
    src_train, src_test = _fleurs(src_lang, use_pred_for_ref, model_name, device)
    tgt_train, tgt_test = _fleurs(tgt_lang, use_pred_for_ref, model_name, device)
    
    # Find common sentence IDs across both languages
    src_train_ids = {sample['id'] for sample in src_train}
    tgt_train_ids = {sample['id'] for sample in tgt_train}
    common_train_ids = src_train_ids.intersection(tgt_train_ids)
    
    src_test_ids = {sample['id'] for sample in src_test}
    tgt_test_ids = {sample['id'] for sample in tgt_test}
    common_test_ids = src_test_ids.intersection(tgt_test_ids)
    
    # Filter each dataset to include only the common sentence IDs
    parallel_train_samples = _prep_parallel_samples(src_train, tgt_train, common_train_ids, use_pred_for_ref, model_name, src_lang, tgt_lang, device)
    parallel_test_samples = _prep_parallel_samples(src_test, tgt_test, common_test_ids, use_pred_for_ref, model_name, src_lang, tgt_lang, device)
    
    return parallel_train_samples, parallel_test_samples

def _prep_parallel_samples(src_data, tgt_data, common_ids, use_pred_for_ref, model_name, src_lang, tgt_lang, device):
    """
    Prepares parallel samples with audio paths and transcriptions for both source and target languages.

    Args:
        src_data (list): Source language data samples.
        tgt_data (list): Target language data samples.
        common_ids (set): Set of common sentence IDs.
        use_pred_for_ref (bool): Whether to use Whisper model predictions for the 'ref' content.
        model_name (str): The name of the Whisper model to use for predictions.
        src_lang (str): The language code for the source audio language.
        tgt_lang (str): The language code for the target audio language.
        device (torch.device): The device to run the Whisper model on.

    Returns:
        list: List of dictionaries containing 'audio', 'ref', 'audio_tgt', and 'ref_src' keys.
    """
    whisper_model_src = None
    whisper_model_tgt = None
    if use_pred_for_ref:
        whisper_model_src = WhisperModel(model_name=model_name, task='transcribe', language=src_lang, device=device)
        whisper_model_tgt = WhisperModel(model_name=model_name, task='translate', language=src_lang, device=device)
        print("Generating predictions for parallel dataset...")
    
    src_dict = {sample['id']: sample for sample in src_data if sample['id'] in common_ids}
    tgt_dict = {sample['id']: sample for sample in tgt_data if sample['id'] in common_ids}

    cache_dir_src = os.path.join(CACHE_DIR, src_lang, model_name or "default", "transcribe")
    cache_dir_tgt = os.path.join(CACHE_DIR, tgt_lang, model_name or "default", "translate")
    os.makedirs(cache_dir_src, exist_ok=True)
    os.makedirs(cache_dir_tgt, exist_ok=True)
    
    parallel_samples = []
    for id in tqdm(common_ids, desc="Processing parallel samples"):
        src_sample = src_dict[id]
        tgt_sample = tgt_dict[id]

        audio_path_src = src_sample['audio']
        cache_file_src = os.path.join(cache_dir_src, f"{src_sample['id']}.json")
        ref_src = src_sample['ref']

        if os.path.exists(cache_file_src):
            with open(cache_file_src, 'r') as f:
                ref_src = json.load(f)['ref']
        elif use_pred_for_ref:
            ref_src = whisper_model_src.predict(audio=audio_path_src)
            with open(cache_file_src, 'w') as f:
                json.dump({'ref': ref_src}, f)

        audio_path_tgt = tgt_sample['audio']
        cache_file_tgt = os.path.join(cache_dir_tgt, f"{tgt_sample['id']}.json")
        ref_tgt = tgt_sample['ref']

        if os.path.exists(cache_file_tgt):
            with open(cache_file_tgt, 'r') as f:
                ref_tgt = json.load(f)['ref']
        elif use_pred_for_ref:
            ref_tgt = whisper_model_tgt.predict(audio=audio_path_tgt)
            with open(cache_file_tgt, 'w') as f:
                json.dump({'ref': ref_tgt}, f)
        
        parallel_samples.append(
            {
                'audio': audio_path_src,
                'ref': ref_tgt,
                'audio_tgt': audio_path_tgt,
                'ref_src': ref_src
            }
        )
    return parallel_samples
