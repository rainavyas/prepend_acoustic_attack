import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from tqdm import tqdm
from whisper.audio import load_audio

from .base import AudioBaseAttacker
from src.tools.tools import AverageMeter



class AudioAttack(AudioBaseAttacker):
    '''
       Prepend adversarial attack in audio space -- designed to mute Whisper by maximizing eot token as first generated token
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3, multiple_model_attack=False, attack_init='random'):
        AudioBaseAttacker.__init__(self, attack_args, whisper_model, device, attack_init=attack_init)
        self.audio_attack_model.multiple_model_attack = multiple_model_attack
        self.optimizer = torch.optim.AdamW(self.audio_attack_model.parameters(), lr=lr, eps=1e-8)


    def _loss(self, logits):
        '''
        The (average) negative log probability of the end of transcript token

        logits: Torch.tensor [batch x vocab_size]
        '''
        tgt_id = self._get_tgt_tkn_id()
        sf = nn.Softmax(dim=1)
        log_probs = torch.log(sf(logits))
        tgt_probs = log_probs[:,tgt_id].squeeze()
        return -1*torch.mean(tgt_probs)
    

    def train_step(self, train_loader, epoch, print_freq=25):
        '''
            Run one train epoch - Projected Gradient Descent
        '''
        losses = AverageMeter()

        # switch to train mode
        self.audio_attack_model.train()

        for i, (audio) in enumerate(train_loader):
            audio = audio[0].to(self.device)

            # Forward pass
            logits = self.audio_attack_model(audio, self.whisper_model)[:,-1,:].squeeze(dim=1)
            loss = self._loss(logits)

            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            # print(self.audio_attack_model.audio_attack_segment.grad)
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


    @staticmethod
    def _prep_dl(data, bs=16, shuffle=False):
        '''
        Create batch of audio vectors
        '''

        print('Loading and batching audio files')
        audio_vectors = []
        for d in tqdm(data):
            audio_np = load_audio(d['audio'])
            audio_vector = torch.from_numpy(audio_np)
            audio_vectors.append(audio_vector)
        
        def pad_sequence(tensors, padding_value=0):
            max_length = max(len(tensor) for tensor in tensors)
            padded_tensors = []
            for tensor in tensors:
                padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=padding_value)
                padded_tensors.append(padded_tensor)
            return padded_tensors

        audio_vectors = pad_sequence(audio_vectors)
        audio_vectors = torch.stack(audio_vectors, dim=0)
        ds = TensorDataset(audio_vectors)
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=8)
        return dl


    def train_process(self, train_data, cache_dir):

        fpath = f'{cache_dir}/prepend_attack_models'
        if not os.path.isdir(fpath):
            os.mkdir(fpath)

        train_dl = self._prep_dl(train_data, bs=self.attack_args.bs, shuffle=True)

        for epoch in range(self.attack_args.max_epochs):
            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train_step(train_dl, epoch)

            if epoch==self.attack_args.max_epochs-1 or (epoch+1)%self.attack_args.save_freq==0:
                # save model at this epoch
                if not os.path.isdir(f'{fpath}/epoch{epoch+1}'):
                    os.mkdir(f'{fpath}/epoch{epoch+1}')
                state = self.audio_attack_model.state_dict()
                torch.save(state, f'{fpath}/epoch{epoch+1}/model.th')










            


