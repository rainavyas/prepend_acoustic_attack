# from tqdm import tqdm
# import torch
# import whisper
# from whisper.tokenizer import get_tokenizer

# from src.data.fleurs import _fleurs
# from src.models.whisper import WhisperModel
# from src.attacker.audio_raw.learn_attack import AudioAttack
# from src.attacker.audio_raw.audio_attack_model_wrapper import AudioAttackModelWrapper


# Get average sequence length of references for different datasets
from src.data.fleurs import _fleurs
from src.data.speech import _librispeech, _tedlium, _mgb, _artie
from src.tools.tools import eval_neg_seq_len

data = _librispeech('test_other')
x = [d['ref'] for d in data]
print("librispeech", -1*eval_neg_seq_len(x)) 

data = _tedlium()
x = [d['ref'] for d in data]
print("tedlium", -1*eval_neg_seq_len(x)) 

data = _mgb()
x = [d['ref'] for d in data]
print("mgb", -1*eval_neg_seq_len(x))

data = _artie()
x = [d['ref'] for d in data]
print("artie", -1*eval_neg_seq_len(x)) 



_, data = _fleurs(lang='fr')
x = [d['ref'] for d in data]
print("fleurs-fr", -1*eval_neg_seq_len(x))

_, data = _fleurs(lang='de')
x = [d['ref'] for d in data]
print("fleurs-de", -1*eval_neg_seq_len(x))

_, data = _fleurs(lang='ru')
x = [d['ref'] for d in data]
print("fleurs-ru", -1*eval_neg_seq_len(x))

_, data = _fleurs(lang='ko')
x = [d['ref'] for d in data]
print("fleurs-ko", -1*eval_neg_seq_len(x))




# # fraction of samples that generate transcribe token as the first token with the transcribe acoustic realization attack

# model_name='whisper-small-multi'
# lang = 'fr'
# task = 'translate'
# attack_size = 10240
# device = torch.device('cuda:1')
# attack_model_path = f'experiments/fleurs/{model_name}/{task}/{lang}/attack_train/audio-raw/attack_tokentranscribe/attack_size{attack_size}/clip_val0.02/prepend_attack_models/epoch40/model.th'
# # attack_model_path = 'experiments/librispeech/whisper-tiny/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/epoch40/model.th'

# # load the data
# # _, data = _fleurs(lang)
# _, data = _fleurs(lang)
# dl = AudioAttack._prep_dl(data, bs=8)

# # load the model
# whisper_model = WhisperModel(model_name=model_name, device=device, task=task, language=lang)

# # load the audio attack model wrapper
# audio_attack_model = AudioAttackModelWrapper(whisper_model.tokenizer, attack_size=attack_size, device=device).to(device)
# audio_attack_model.load_state_dict(torch.load(attack_model_path))

# # forward pass for whole dataset and get fraction matches with tgt_id
# tgt_id = whisper_model.tokenizer.transcribe
# matches = 0
# all_pred_ids = []
# for i, (mels) in enumerate(dl):
#     # print(i/len(dl))
#     with torch.no_grad():
#         mels = mels[0].to(device)
#         logits = audio_attack_model(mels, whisper_model)[:,-1,:].squeeze(dim=1)
#         pred_ids = torch.argmax(logits, dim=1)
#         pred_ids = pred_ids.detach().cpu().tolist()
#         # breakpoint()
#         for pred in pred_ids:
#             if pred == tgt_id:
#                 matches +=1
#         all_pred_ids += pred_ids

# print()
# print(matches/len(data))
# print()
# print(all_pred_ids)