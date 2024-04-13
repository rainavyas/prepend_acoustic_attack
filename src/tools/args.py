import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='whisper-small', help='ASR model')
    commandLineParser.add_argument('--task', type=str, default='transcribe', choices=['transcribe', 'translate'], help='Whisper task. N.b. translate is only X-en')
    commandLineParser.add_argument('--language', type=str, default='en', help='Source audio language')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--data_name', type=str, default='librispeech', help='dataset for exps; for flores: flores-english-french')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    return commandLineParser.parse_known_args()

def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    # train attack args
    commandLineParser.add_argument('--attack_method', type=str, default='audio-raw', choices=['audio-raw', 'audio-tonal', 'mel'], help='Adversarial attack approach for training')
    commandLineParser.add_argument('--max_epochs', type=int, default=20, help='Training epochs for attack')
    commandLineParser.add_argument('--save_freq', type=int, default=1, help='Epoch frequency for saving attack')
    commandLineParser.add_argument('--attack_size', type=int, default=5120, help='Length of attack segment')
    commandLineParser.add_argument('--bs', type=int, default=16, help='Batch size for training attack')
    commandLineParser.add_argument('--lr', type=float, default=1e-3, help='Adversarial Attack learning rate')
    commandLineParser.add_argument('--clip_val', type=float, default=-1, help='Value (maximum) to clip the log mel vectors. -1 means no clipping')




    # eval attack args
    commandLineParser.add_argument('--attack_epoch', type=int, default=-1, help='Specify which training epoch of attack to evaluate; -1 means no attack')
    commandLineParser.add_argument('--force_run', action='store_true', help='Do not load from cache')
    commandLineParser.add_argument('--not_none', action='store_true', help='Do not evaluate the none attack')
    commandLineParser.add_argument('--eval_train', action='store_true', help='Evaluate attack on the train split')


    # eval attack args for attack transferability
    commandLineParser.add_argument('--transfer', action='store_true', help='Indicate it is a transferability attack (across model or dataset) for mel whitebox attack')
    commandLineParser.add_argument('--attack_model_dir', type=str, default='', help='path to trained attack to evaluate')
    return commandLineParser.parse_known_args()

def analysis_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--compare_with_audio', action='store_true', help='Include a real audio file')
    commandLineParser.add_argument('--sample_id', type=int, default=42, help='Specify which data sample to compare to')
    return commandLineParser.parse_known_args()


