import torch
from nemo.collections.asr.models import EncDecMultiTaskModel
import json
import os

os.environ['NEMO_CACHE_DIR'] = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/data/cache'

class CanaryModel:
    def __init__(self, device=torch.device('cpu'), task='transcribe', language='en', pnc=True):
        self.model_name = 'canary'
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b').to(device)
        self.tokenizer = self.model.tokenizer
        self.taskname = 'asr' if task == 'transcribe' else 's2t_translation'
        self.task = task
        if task == 'transcribe':
            self.src_lang = language.split('_')[0]
            self.tgt_lang = self.src_lang
        else:
            self.src_lang = language.split('_')[0]
            self.tgt_lang = language.split('_')[1]
        self.pnc = 'yes' if pnc else 'no'
        
        # Update decode params
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(decode_cfg)

        self.prep_sot_ids()
    
    def prep_sot_ids(self):
        '''
            Special Tokens used by decoder
            <|startoftranscript|><|source_lang|><|taskname|><|target_lang|><|pnc|>
        '''
        spl_tkns = self.tokenizer.special_tokens
        ids = []
        ids.append(spl_tkns['<|startoftranscript|>'])
        ids.append(spl_tkns[f'<|{self.src_lang}|>'])
        ids.append(spl_tkns[f'<|{self.task}|>']) # transcribe or translate
        ids.append(spl_tkns[f'<|{self.tgt_lang}|>'])
        if self.pnc == 'yes':
            ids.append(spl_tkns['<|pnc|>'])
        else:
            ids.append(spl_tkns['<|nopnc|>'])
        
        self.sot_ids = ids


    def create_manifest(self, audio_path):
        '''
            Create input manifest file for the Canary model
        '''
        manifest_entry = {
            "audio_filepath": audio_path,
            "duration": None,  # Duration can be set to None
            "taskname": self.taskname,
            "source_lang": self.src_lang,
            "target_lang": self.tgt_lang,
            "pnc": self.pnc,
            "answer": "na"
        }
        
        manifest_path = 'experiments/input_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entry, f)
            f.write('\n')
        
        return manifest_path

    def predict(self, audio='', **kwargs):
        '''
            Run through Canary model
        '''
        # Create input manifest file
        manifest_path = self.create_manifest(audio)
        
        # Pass through Canary model
        predicted_text = self.model.transcribe(
            manifest_path,
            batch_size=1,
            verbose=False
        )
        return predicted_text[0] if predicted_text else None


