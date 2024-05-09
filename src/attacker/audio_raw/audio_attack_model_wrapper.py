import torch
import torch.nn as nn
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES, load_audio

class AudioAttackModelWrapper(nn.Module):
    '''
        Whisper Model wrapper with learnable audio segment attack prepended to speech signals
    '''
    def __init__(self, tokenizer, attack_size=5120, device=None):
        super(AudioAttackModelWrapper, self).__init__()
        self.attack_size = attack_size
        self.audio_attack_segment = nn.Parameter(torch.rand(attack_size))
        self.tokenizer = tokenizer
        self.device = device
        self.multiple_model_attack = False
    
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
        if self.multiple_model_attack:
            n_mels = whisper_model.models[0].dims.n_mels
        else:
            n_mels = whisper_model.model.dims.n_mels
        padded_mel = log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(padded_mel, N_FRAMES)
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
        if self.multiple_model_attack:
            # pass through each target model
            sf = nn.Softmax(dim=-1)
            pred_probs = []
            for model in whisper_model.models:
                pred_probs.append(sf(model.forward(mel, sot_ids)))
            return torch.mean(torch.stack(pred_probs), dim=0) 
        else:
            return whisper_model.model.forward(mel, sot_ids)
    
    def transcribe(self,
        whisper_model,
        audio,
        do_attack=True,
        without_timestamps=False
    ):

        '''
            Mimics the original Whisper transcribe functions but prepends the adversarial attack
            in the audio space

                do_attack parameter is a boolean to do the attack or not
        '''
        if do_attack:
            # prepend attack
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio).to(self.device)
            audio = torch.cat((self.audio_attack_segment, audio), dim=0)

        return whisper_model.predict(audio, without_timestamps=without_timestamps)


