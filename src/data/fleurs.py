from datasets import load_dataset

LANG_MAPPER = {
    'fr'    :   'fr_fr',
    'de'    :   'de_de',
    'ru'    :   'ru_ru',
    'ko'    :   'ko_kr',
    'en'    :   'en_us'
}

def _fleurs(lang='fr'):
    """
    Loads the FLEURS dataset for a given language or language pair.

    Args:
        lang (str): The language code or language pair code in the format 'src_tgt'.

    Returns:
        tuple: Parallel training and test samples if a language pair is specified,
               otherwise training and test samples for the specified language.
    """
    if '_' in lang:
        return _fleurs_parallel(lang)
    else:
        train_data = load_dataset("google/fleurs", f"{LANG_MAPPER[lang]}", split="train")
        test_data = load_dataset("google/fleurs", f"{LANG_MAPPER[lang]}", split="test")
        return _prep_samples(train_data), _prep_samples(test_data)

def _prep_samples(data):
    """
    Prepares samples with audio paths and transcriptions.

    Args:
        data (Dataset): The dataset to prepare samples from.

    Returns:
        list: List of dictionaries containing 'ref' and 'audio' keys.
    """
    samples = []
    for sample in list(data):
        samples.append(
            {
                'id'    :   sample['id'],
                'ref'   :   sample['transcription'],
                'audio' :   '/'.join(sample['path'].split('/')[:-1]) + '/' + sample['audio']['path']
            }
        )
    return samples

def _fleurs_parallel(lang_pair='fr_en'):
    """
    Loads parallel sentences between the specified source and target languages from the FLEURS dataset.

    This function prepares the dataset for a spoken language translation task, where each sample contains:
    - 'audio': Audio of the source language.
    - 'ref': Transcription of the target language.
    - 'audio_tgt': Audio of the target language.
    - 'ref_src': Transcription of the source language.

    Args:
        lang_pair (str): The language pair code in the format 'src_tgt'.

    Returns:
        tuple: Parallel training and test samples.
    """
    src_lang, tgt_lang = lang_pair.split('_')
    
    # Load datasets for source and target languages
    src_train, src_test = _fleurs(src_lang)
    tgt_train, tgt_test = _fleurs(tgt_lang)
    
    # Find common sentence IDs across both languages
    src_train_ids = {sample['id'] for sample in src_train}
    tgt_train_ids = {sample['id'] for sample in tgt_train}
    common_train_ids = src_train_ids.intersection(tgt_train_ids)
    
    src_test_ids = {sample['id'] for sample in src_test}
    tgt_test_ids = {sample['id'] for sample in tgt_test}
    common_test_ids = src_test_ids.intersection(tgt_test_ids)
    
    # Filter each dataset to include only the common sentence IDs
    parallel_train_samples = _prep_parallel_samples(src_train, tgt_train, common_train_ids)
    parallel_test_samples = _prep_parallel_samples(src_test, tgt_test, common_test_ids)
    
    return parallel_train_samples, parallel_test_samples

def _prep_parallel_samples(src_data, tgt_data, common_ids):
    """
    Prepares parallel samples with audio paths and transcriptions for both source and target languages.

    Args:
        src_data (list): Source language data samples.
        tgt_data (list): Target language data samples.
        common_ids (set): Set of common sentence IDs.

    Returns:
        list: List of dictionaries containing 'audio', 'ref', 'audio_tgt', and 'ref_src' keys.
    """
    src_dict = {sample['id']: sample for sample in src_data if sample['id'] in common_ids}
    tgt_dict = {sample['id']: sample for sample in tgt_data if sample['id'] in common_ids}
    
    parallel_samples = []
    for id in common_ids:
        src_sample = src_dict[id]
        tgt_sample = tgt_dict[id]
        parallel_samples.append(
            {
                'audio'         :   src_sample['audio'],
                'ref'           :   tgt_sample['ref'],
                'audio_tgt'     :   tgt_sample['audio'],
                'ref_src'       :   src_sample['ref']
            }
        )
    return parallel_samples
