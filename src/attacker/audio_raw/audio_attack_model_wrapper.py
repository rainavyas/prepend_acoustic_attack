import torch
import torch.nn as nn
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES

class AudioAttackModelWrapper(nn.Module):
    '''
        Whisper Model wrapper with learnable audio segment attack prepended to speech signals
    '''
    def __init__(self, attack_size=5120, device=None):
        super(AudioAttackModelWrapper, self).__init__()
        self.attack_size = attack_size
        self.audio_attack_segment = nn.Parameter(torch.rand(attack_size)) 
    
    def forward(self, audio_vector, whisper_model):
        '''
            audio_vector: Torch.tensor: [Batch x Audio Length]
            whisper_model: encoder-decoder model

            Returns the logits for the first transcribed token
        '''
        # prepend attack segment
        X = self.audio_attack_segment.unsqueeze(0).expand(audio_vector.size(0), -1)
        attacked_audio_vector = torch.cat((X, audio_vector), dim=1)

        # forward pass through full model
        mel = self._audio_to_mel(attacked_audio_vector, whisper_model)
        return self._mel_to_logit(mel, whisper_model)
    

    def _audio_to_mel(self, audio: torch.Tensor, whisper_model):
        '''
            audio: [Batch x Audio length]
            based on https://github.com/openai/whisper/blob/main/whisper/audio.py
        '''
        padded_mel = log_mel_spectrogram(audio, whisper_model.model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES)
        return mel
    
    def _mel_to_logit(self, mel: torch.Tensor, whisper_model):
        '''
            Forward pass through the whisper model of the mel vectors
            expect mel vectors passed as a batch and padded to 30s of audio length
            mel: torch.Tensor [B x dim x num_vectors]
        '''
        # create batch of start of transcript tokens
        sot_ids = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps)
        sot_ids = sot_ids.to(self.device)
        sot_ids = sot_ids.unsqueeze(0).expand(mel.size(0), -1)
        return whisper_model.model.forward(mel, sot_ids)
    
    def transcribe(self,
        whisper_model,
        audio,
        do_attack=True
    ):

        '''
            Mimics the original Whisper transcribe functions but prepends the adversarial attack
            in the audio space

                do_attack parameter is a boolean to do the attack or not
        '''
        if do_attack:
            # prepend attack
            audio = load_audio(audio)
            audio = torch.from_numpy(audio)
            audio = torch.cat((self.audio_attack_segment, audio), dim=0)

        return whisper_model.predict(audio)

