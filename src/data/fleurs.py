from datasets import load_dataset

LANG_MAPPER = {
    'fr'    :   'fr_fr',
    'de'    :   'de_de',
    'ru'    :   'ru_ru',
    'ko'    :   'ko_kr'
}

def _fleurs(lang='fr'):
    val_data = load_dataset("google/fleurs", f"{LANG_MAPPER[lang]}", split="validation")
    test_data = load_dataset("google/fleurs", f"{LANG_MAPPER[lang]}", split="test")
    return _prep_samples(val_data), _prep_samples(test_data)

def _prep_samples(data):
    samples = []
    for sample in list(data):
        samples.append(
            {
                'ref'   :   sample['transcription'],
                'audio' :   '/'.join(sample['path'].split('/')[:-1]) + '/' + sample['audio']['path']
            }
        )
    return samples