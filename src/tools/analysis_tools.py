import torch
from whisper.audio import load_audio

def saliency(audio, audio_attack_model, whisper_model, device):
    '''
        Get saliency of audio and audio_attack_segment
    '''
    adv_grad, audio_grad = _saliency_calculation(audio, audio_attack_model, whisper_model, device)

    adv_grad_norm = torch.linalg.vector_norm(adv_grad)
    audio_grad_norm = torch.linalg.vector_norm(audio_grad)

    if len(audio) == 0:
        audio_grad_norm = 0
    else:
        audio_grad_norm = audio_grad_norm.detach().cpu().item()

    return adv_grad_norm.detach().cpu().item(), audio_grad_norm


def frame_level_saliency(audio, audio_attack_model, whisper_model, device):
    '''
        get the saliency per frame of attack segment and speech signal
    '''
    adv_grad, audio_grad = _saliency_calculation(audio, audio_attack_model, whisper_model, device)
    saliencies = torch.abs(torch.cat((adv_grad, audio_grad), dim=0))
    return saliencies.detach().cpu()

def _saliency_calculation(audio, audio_attack_model, whisper_model, device):
    '''
        Forward-backward pass
    '''
    if isinstance(audio, str):
        audio = load_audio(audio)
    audio = torch.from_numpy(audio).to(device)
    audio.requires_grad = True
    audio.retain_grad()

    audio_attack_model.eval()

    # forward pass
    logits = audio_attack_model.forward(audio.unsqueeze(dim=0), whisper_model)[:,-1,:].squeeze(dim=1).squeeze(dim=0)
    sf = torch.nn.Softmax(dim=0)
    probs = sf(logits)
    pred_class = torch.argmax(probs)
    prob = probs[pred_class]

    # compute gradients
    prob.backward()
    adv_grad = audio_attack_model.audio_attack_segment.grad
    audio_grad = audio.grad

    return adv_grad, audio_grad


def get_decoder_proj_mat(whisper_model):
    '''
    Extract the final projection matrix used in the Whisper decoder to obtain the logits

    N.B. this projection matrix is the same as the token id to embedding matrix used in the input to the decoder
        This is standard for the Transformer decoder (refer to the attention is all you need paper)

    Should return W: Tensor [V x k]
        V = vocabulary size
        k = embedding size
    '''
    W = whisper_model.decoder.token_embedding.weight
    return W

def get_rel_pos(W):
    '''
        Return the similarity of each row vector (normalized) with every other row vector

        W: Tensor [V x k]

        Return W^TW: Tensor [V x V]
    '''
    W_norm = W / W.norm(dim=1, keepdim=True)
    # Return the similarity matrix
    return torch.matmul(W_norm, W_norm.T)