from .speech import _librispeech, _tedlium, _mgb, _artie
from .fleurs import _fleurs


def load_data(core_args):
    '''
        Return data as train_data, test_data
        Each data is a list (over data samples), where each sample is a dictionary
            sample = {
                        'audio':    <path to utterance audio file>,
                        'ref':      <Reference transcription>,
                    }
    '''
    if core_args.data_name == 'fleurs':
        return _fleurs(lang=core_args.language)

    if core_args.data_name == 'tedlium':
        return None, _tedlium()

    if core_args.data_name == 'mgb':
        return None, _mgb()

    if core_args.data_name == 'artie':
        return None, _artie()

    if core_args.data_name == 'librispeech':
        return _librispeech('dev_other'), _librispeech('test_other')

    


