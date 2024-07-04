import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from tqdm import tqdm
from whisper.audio import load_audio

from .learn_attack_hallucinate import AudioAttackHallucinate
from src.tools.tools import set_seeds, AverageMeter



class AudioAttackTranslate(AudioAttackHallucinate):
    '''
       Prepend adversarial attack in audio space -- designed to make Whisper always translate (to English) even if sot tokens request transcription
       Assume that data prepared such that it has 'audio' in source language and transcription as 'ref' in English (tgt language)
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3, multiple_model_attack=False, attack_init='random'):
        AudioAttackHallucinate.__init__(self, attack_args, whisper_model, device, lr=lr, multiple_model_attack=multiple_model_attack, attack_init=attack_init)
        self.max_length = 400


    def _loss(self, input_ids, logits, seq_len):
        '''
        Teacher forced cross-entropy loss for Transformer decoder

        input_ids: torch.Tensor: [batch x max_len_sequence]
            Input ids to decoder (padded with zeros)

        seq_len: torch.Tensor: [batch]
            Length of each sequence in the batch for input_ids
        
        logits: torch.Tensor: [batch x (self.audio_attack_model.len_sot_ids + max_len_sequence) x vocab_size]
            Predicted logits from Transformer decoder with (sot_ids, input_ids) at the input of the decoder 
        

        
        Assume that input_ids and seq_len do not account for the starting tokens, the length of which can be given by
            self.audio_attack_model.len_sot_ids

        '''
        # Get the length of the starting tokens
        len_sot_ids = self.audio_attack_model.len_sot_ids
        
        # Shift logits to match the input_ids
        shifted_logits = logits[:, len_sot_ids-1:-1, :]

        # Flatten logits and targets for cross-entropy
        batch_size, max_len_sequence = input_ids.size()
        vocab_size = logits.size(-1)
        
        # Create a mask based on sequence lengths
        mask = torch.arange(max_len_sequence, device=self.device).expand(batch_size, max_len_sequence) < seq_len.unsqueeze(1)
        
        # Flatten the mask
        mask = mask.view(-1)
        
        # Compute the loss
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, vocab_size),
            input_ids.reshape(-1),
            reduction='none'
        )
        
        # Apply the mask
        loss = loss * mask
        
        # Sum the losses and divide by the number of non-padded tokens
        loss = loss.sum() / mask.sum().float()
        
        return loss

    

    def train_step(self, train_loader, epoch, print_freq=25):
        '''
            Run one train epoch - Projected Gradient Descent
        '''
        losses = AverageMeter()

        # switch to train mode
        self.audio_attack_model.train()

        for i, (audio, decoder_input, seq_len) in enumerate(train_loader):
            audio = audio.to(self.device)
            decoder_input = decoder_input.to(self.device)
            seq_len = seq_len.to(self.device)

            # Forward pass
            logits = self.audio_attack_model(audio, self.whisper_model, decoder_input=decoder_input)
            loss = self._loss(decoder_input, logits, seq_len)

            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.attack_args.clip_val != -1:
                max_val = self.attack_args.clip_val
            else:
                max_val = 100000
            with torch.no_grad():  
                self.audio_attack_model.audio_attack_segment.clamp_(min=-1*max_val, max=max_val)
        
            # record loss
            losses.update(loss.item(), audio.size(0))
            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.5f} ({losses.avg:.5f})')        











            


