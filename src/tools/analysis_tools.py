import torch
from whisper.audio import load_audio
import string

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

def get_rel_pos(W, real_token_ids, device=torch.device('cpu')):
    '''
    Return the similarity of each row vector (normalized) with every other row vector,
    where each row (r) vector is the rel_pos_vector for target token r.

    W: Tensor [V x k]
    real_token_ids: List[int] - List of real acoustic token ids

    Return rel_pos_matrix: Tensor [V x V]
    '''
    V = W.shape[0]
    W_norm = W / W.norm(dim=1, keepdim=True)

    # Compute the dot product for all tokens
    rel_pos_matrix = torch.matmul(W_norm, W_norm.T).cpu()

    # Zero out the columns that are not part of the real acoustic token ids
    mask = torch.zeros(V, V)
    mask[:, real_token_ids] = 1
    rel_pos_matrix = rel_pos_matrix * mask

    return rel_pos_matrix


def get_real_acoustic_token_ids(tokenizer, vocab_size):
    '''
    Identify real acoustic token ids based on the criteria that the token begins with a
    letter in the English alphabet or a numeral 1-9.

    tokenizer: Tokenizer object
    vocab_size: int - The size of the vocabulary

    Return real_token_ids: List[int]
    '''
    real_token_ids = []
    for token_id in range(vocab_size):
        token = tokenizer.decode([token_id])
        if token and is_real_acoustic_token(token):
            real_token_ids.append(token_id)
    return real_token_ids

def is_real_acoustic_token(token):
    ''' Check if the token begins with a letter in the English alphabet or a numeral 1-9. '''
    return token[0] in string.ascii_letters or token[0] in '123456789'