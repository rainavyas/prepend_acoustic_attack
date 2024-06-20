import torch
import torch.nn as nn
from whisper.audio import load_audio

def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask

class AudioAttackCanaryModelWrapper(nn.Module):
    '''
        Canary Model wrapper with learnable audio segment attack prepended to speech signals
    '''
    def __init__(self, tokenizer, attack_size=5120, device=None, attack_init='random'):
        super(AudioAttackCanaryModelWrapper, self).__init__()
        self.attack_size = attack_size
        self.tokenizer = tokenizer
        self.device = device

        self.len_sot_ids = 5 # always 5 for canary model

        if attack_init == 'random':
            self.audio_attack_segment = nn.Parameter(torch.rand(attack_size))
        else:
            # load init attack from attack_init path
            loaded_params = torch.load(attack_init)
            if 'audio_attack_segment' in loaded_params:
                initial_value = loaded_params['audio_attack_segment']
                self.audio_attack_segment = nn.Parameter(initial_value.to(device))
            else:
                raise ValueError("Invalid attack_init path provided.")
    
    def forward(self, audio_vector, canary_model, decoder_input=None):
        '''
            audio_vector: Torch.tensor: [Batch x Audio Length]
            canary_model: encoder (conformer) - Transformer decoder model

            Returns the logits
        '''
        # prepend attack segment
        X = self.audio_attack_segment.unsqueeze(0).expand(audio_vector.size(0), -1)
        attacked_audio_vector = torch.cat((X, audio_vector), dim=1)

        # Forward pass through preprocessor
        processed_signal, processed_signal_length = canary_model.preprocessor(
            input_signal=attacked_audio_vector, length=torch.tensor([attacked_audio_vector.size(1)] * attacked_audio_vector.size(0))
        )

        # Forward pass through encoder
        encoded, encoded_len = canary_model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        # Project encoder outputs if necessary
        enc_states = encoded.permute(0, 2, 1)
        enc_states = canary_model.encoder_decoder_proj(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if canary_model.use_transf_encoder:
            enc_states = canary_model.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        # Ensure the decoder input starts with the appropriate start-of-transcript tokens
        sot_ids = torch.tensor(self.sot_ids).to(self.device)
        decoder_input_ids = sot_ids.unsqueeze(0).expand(enc_states.size(0), -1)
        
        if decoder_input is not None:
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_input), dim=1)

        # Create decoder mask
        dec_mask = lens_to_mask(decoder_input_ids.size(1), decoder_input_ids.size(1)).to(decoder_input_ids.dtype)
        dec_states = canary_model.transf_decoder(
            input_ids=decoder_input_ids, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
        )
        transf_log_probs = canary_model.log_softmax(hidden_states=dec_states)
        return transf_log_probs
    
    
    def transcribe(self,
        canary_model,
        audio,
        do_attack=True,
    ):

        '''
            Mimics the original Canary transcribe functions but prepends the adversarial attack
            in the audio space

                do_attack parameter is a boolean to do the attack or not
        '''
        if do_attack:
            # prepend attack
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio).to(self.device)
            audio = torch.cat((self.audio_attack_segment, audio), dim=0)

        return canary_model.predict(audio) # To change to pass audio as altered filepath


