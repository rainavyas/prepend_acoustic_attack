import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from tqdm import tqdm
from whisper.audio import load_audio

from .learn_attack import AudioAttack
from src.tools.tools import set_seeds, AverageMeter



class AudioAttackHallucinate(AudioAttack):
    '''
       Prepend adversarial attack in audio space -- designed to make Whisper hallucinate by minimizing eot token prediction
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3, multiple_model_attack=False, attack_init='random'):
        AudioAttack.__init__(self, attack_args, whisper_model, device, lr=lr, multiple_model_attack=multiple_model_attack, attack_init=attack_init)
        self.max_length = 400


    def _loss(self, logits, seq_len):
        '''
        The (average) log probability of the end of transcript token

        logits: Torch.tensor [batch x seq_len x vocab_size]
        seq_len: Torch.tensor [batch]
        '''
        tgt_id = self._get_tgt_tkn_id()

        # Compute log probabilities over the vocabulary dimension
        sf = nn.Softmax(dim=2)
        log_probs = torch.log(sf(logits))
        
        # Gather the log probabilities for the target positions and target token
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        tgt_probs = log_probs[batch_indices, seq_len-1, tgt_id]


        return -1/torch.mean(tgt_probs)
    

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
            loss = self._loss(logits, seq_len + self.audio_attack_model.len_sot_ids)

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


    def _pad_sequence(self, tensors, padding_value=0):
        max_length = max(len(tensor) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=padding_value)
            padded_tensors.append(padded_tensor)
        return padded_tensors

    def _prep_dl(self, data, bs=4, shuffle=False):
        '''
        Create batches of audio vectors, token IDs, and text lengths
        '''
        
        print('Loading and batching audio files and ref token IDs')
        audio_vectors = []
        texts = []
        
        print('audio loading')
        for d in tqdm(data):
            audio_np = load_audio(d['audio'])
            audio_vector = torch.from_numpy(audio_np)
            audio_vectors.append(audio_vector)
            texts.append(d['ref'])
        
        audio_vectors = self._pad_sequence(audio_vectors)
        audio_vectors = torch.stack(audio_vectors, dim=0)
        
        # Tokenize texts manually, ensuring padding and truncation
        tokenized_texts = []
        text_lengths = []
        print('text tokenization')
        for text in tqdm(texts):
            if self.whisper_model.model_name == 'canary':
                token_ids = self.whisper_model.tokenizer.text_to_ids(text, 'en')[:self.max_length] # assuming reference text is English
            else:
                token_ids = self.whisper_model.tokenizer.encode(text)[:self.max_length]  
            text_lengths.append(len(token_ids))  # Original length before padding
            if len(token_ids) < self.max_length:
                token_ids.extend([0] * (self.max_length - len(token_ids)))  # Pad
            tokenized_texts.append(torch.tensor(token_ids))

        text_token_ids = torch.stack(tokenized_texts, dim=0)
        text_lengths = torch.tensor(text_lengths)
        
        ds = TensorDataset(audio_vectors, text_token_ids, text_lengths)
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        
        return dl











            


