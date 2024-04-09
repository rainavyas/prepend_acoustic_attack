import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from tqdm import tqdm


from .base import MelBaseAttacker
from src.tools.tools import set_seeds, AverageMeter



class SoftPromptAttack(MelBaseAttacker):
    '''
        Soft-prompting style adversarial attack in mel-vector space

        Learn a short sequence of mel-vectors pre-pended to the input-audio mel-vectors to maximise the probability of the end-of-transcript token
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3):
        MelBaseAttacker.__init__(self, attack_args, whisper_model, device)
        self.optimizer = torch.optim.AdamW(self.softprompt_model.parameters(), lr=lr, eps=1e-8)

    def _loss(self, logits):
        '''
        The (average) negative log probability of the end of transcript token

        logits: Torch.tensor [batch x vocab_size]
        '''
        eot_id = self.tokenizer.eot

        sf = nn.Softmax(dim=1)
        log_probs = torch.log(sf(logits))
        eot_probs = log_probs[:,eot_id].squeeze()
        return -1*torch.mean(eot_probs)
    
    # def _regularization(self, softprompt):
    #     '''
    #         Regularization to ensure temporal smoothness of the adversarial softprompt vectors
    #     '''
    #     weighting = 0.1
    #     return weighting*torch.sum(torch.abs(torch.diff(softprompt)))


    def train_step(self, train_loader, epoch, print_freq=25):
        '''
            Run one train epoch
        '''
        losses = AverageMeter()

        # switch to train mode
        self.softprompt_model.train()

        for i, (mels) in enumerate(train_loader):
            mels = mels[0].to(self.device)

            # Forward pass
            logits = self.softprompt_model(mels, self.whisper_model)[:,-1,:].squeeze(dim=1)
            loss_main = self._loss(logits)
            # loss_reg =  self._regularization(self.softprompt_model.softprompt)
            # loss = loss_main + loss_reg
            loss = loss_main

            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.attack_args.clip_val != -1:
                with torch.no_grad():  
                    self.softprompt_model.softprompt.clamp_(max=self.attack_args.clip_val)

            # record loss
            losses.update(loss.item(), mels.size(0))
            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.5f} ({losses.avg:.5f})')
        

    def _prep_dl(self, data, bs=16, shuffle=False):
        '''
        Create batch of mel vectors
        '''

        print('Creating mel vectors from audio files')
        mels = []
        for d in tqdm(data):
            mels.append(self.audio_to_mel(d['audio']))

        mels = torch.stack(mels, dim=0)
        ds = TensorDataset(mels)
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        return dl


    def train_process(self, train_data, cache_dir):
        set_seeds(1)

        fpath = f'{cache_dir}/softprompt_models'
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
                state = self.softprompt_model.state_dict()
                torch.save(state, f'{fpath}/epoch{epoch+1}/model.th')










            


