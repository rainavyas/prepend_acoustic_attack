from datasets import load_dataset

LANG_MAPPER = {
    'fr'    :   'fr_fr',
    'de'    :   'de_de',
    'ru'    :   'ru_ru',
    'ko'    :   'ko_kr'
}

def _fleurs(lang='fr'):
    data = load_dataset("google/fleurs", f"{LANG_MAPPER[lang]}", split="test")
    samples = []
    for sample in list(data):
        samples.append(
            {
                'ref'   :   sample['transcription'],
                # 'audio' :   sample['audio']['array']
                'audio' :   '/'.join(sample['path'].split('/')[:-1]) + '/' + sample['audio']['path']
            }
        )
    return samples