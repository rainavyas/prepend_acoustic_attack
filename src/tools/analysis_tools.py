import torch
from whisper.audio import load_audio

def saliency(audio, audio_attack_model, whisper_model, device):
    '''
        Get saliency of audio and audio_attack_segment
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
    # adv_grad_norm = torch.linalg.vector_norm(adv_grad)/(len(adv_grad)**0.5)
    adv_grad_norm = torch.linalg.vector_norm(adv_grad)

    audio_grad = audio.grad
    # audio_grad_norm = torch.linalg.vector_norm(audio_grad)/(len(audio_grad)**0.5)
    audio_grad_norm = torch.linalg.vector_norm(audio_grad)

    if len(audio) == 0:
        audio_grad_norm = 0
    else:
        audio_grad_norm = audio_grad_norm.detach().cpu().item()


    return adv_grad_norm.detach().cpu().item(), audio_grad_norm