import torch
import whisper
import editdistance
from whisper.tokenizer import get_tokenizer


CACHE_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/experiments/rm2114/.cache'

MODEL_NAME_MAPPER = {
    'whisper-tiny'  : 'tiny.en',
    'whisper-tiny-multi'  : 'tiny',
    'whisper-base'  : 'base.en',
    'whisper-base-multi'  : 'base',
    'whisper-small' : 'small.en',
    'whisper-small-multi' : 'small',
    'whisper-medium'  : 'medium.en',
    'whisper-medium-multi'  : 'medium',
    'whisper-large'  : 'large',
}

class WhisperModel:
    '''
        Wrapper for Whisper ASR Transcription
    '''
    def __init__(self, model_name='whisper-small', device=torch.device('cpu'), task='transcribe', language='en'):
        self.model = whisper.load_model(MODEL_NAME_MAPPER[model_name], device=device, download_root=CACHE_DIR)
        self.task = task
        self.language = language # source audio language
        self.tokenizer = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages, language=self.language, task=self.task)

    
    def predict(self, audio='', initial_prompt=None, without_timestamps=False):
        '''
            Whisper decoder output here
        '''
        result = self.model.transcribe(audio, language=self.language, task=self.task, initial_prompt=initial_prompt, without_timestamps=without_timestamps)
        segments = []
        for segment in result['segments']:
            segments.append(segment['text'].strip())
        return ' '.join(segments)

