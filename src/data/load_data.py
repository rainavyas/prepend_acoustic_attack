from .speech import _librispeech, _tedlium, _mgb, _artie
from .fleurs import _fleurs
from src.tools.tools import get_default_device


def load_data(core_args):
    '''
        Return data as train_data, test_data
        Each data is a list (over data samples), where each sample is a dictionary
            sample = {
                        'audio':    <path to utterance audio file>,
                        'ref':      <Reference transcription>,
                    }
    '''
    def load_single_dataset(data_name):
        if data_name == 'fleurs':
            device = get_default_device(core_args.gpu_id)
            return _fleurs(lang=core_args.language, use_pred_for_ref=core_args.use_pred_for_ref, model_name=core_args.model_name[0], device=device)
        elif data_name == 'tedlium':
            return None, _tedlium()
        elif data_name == 'mgb':
            return None, _mgb()
        elif data_name == 'artie':
            return None, _artie()
        elif data_name == 'librispeech':
            return _librispeech('dev_other'), _librispeech('test_other')
        else:
            raise ValueError(f"Unknown dataset name: {data_name}")

    if isinstance(core_args.data_name, list) and len(core_args.data_name) > 1:
        train_data_combined, test_data_combined = [], []
        for data_name in core_args.data_name:
            train_data, test_data = load_single_dataset(data_name)
            if train_data is not None:
                train_data_combined.extend(train_data)
            if test_data is not None:
                test_data_combined.extend(test_data)
        return train_data_combined, test_data_combined
    else:
        # If data_name is a single string or a list with one element
        return load_single_dataset(core_args.data_name[0] if isinstance(core_args.data_name, list) else core_args.data_name)



# def load_data(core_args):
#     '''
#         Return data as train_data, test_data
#         Each data is a list (over data samples), where each sample is a dictionary
#             sample = {
#                         'audio':    <path to utterance audio file>,
#                         'ref':      <Reference transcription>,
#                     }
#     '''
#     if core_args.data_name == 'fleurs':
#         device = get_default_device(core_args.gpu_id)
#         return _fleurs(lang=core_args.language, use_pred_for_ref=core_args.use_pred_for_ref, model_name=core_args.model_name[0], device=device)

#     if core_args.data_name == 'tedlium':
#         return None, _tedlium()

#     if core_args.data_name == 'mgb':
#         return None, _mgb()

#     if core_args.data_name == 'artie':
#         return None, _artie()

#     if core_args.data_name == 'librispeech':
#         return _librispeech('dev_other'), _librispeech('test_other')

    


