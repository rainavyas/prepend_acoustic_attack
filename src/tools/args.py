import argparse


def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument(
        "--model_name",
        type=str,
        default="whisper-small",
        nargs="+",
        help="ASR model. Can pass multiple models if multiple models to be loaded",
    )
    commandLineParser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task. N.b. translate is only X-en",
    )
    commandLineParser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Source audio language or if performing machine translation do something like fr_en",
    )
    commandLineParser.add_argument(
        "--gpu_id", type=int, default=0, help="select specific gpu"
    )
    commandLineParser.add_argument(
        "--data_name",
        type=str,
        default="librispeech",
        nargs="+",
        help="dataset for exps;",
    )
    commandLineParser.add_argument(
        "--use_pred_for_ref", action="store_true", help="Implemented for Fleurs dataset. Use model predictions for the reference transcriptions."
    )
    commandLineParser.add_argument("--seed", type=int, default=1, help="select seed")
    commandLineParser.add_argument(
        "--force_cpu", action="store_true", help="force cpu use"
    )
    return commandLineParser.parse_known_args()


def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    # train attack args
    commandLineParser.add_argument(
        "--attack_method",
        type=str,
        default="audio-raw",
        choices=["audio-raw", "mel"],
        help="Adversarial attack approach for training",
    )
    commandLineParser.add_argument(
        "--attack_token",
        type=str,
        default="eot",
        choices=["eot", "transcribe"],
        help="Which non-acoustic token are we learning an acoustic realization for.",
    )
    commandLineParser.add_argument(
        "--attack_command",
        type=str,
        default="mute",
        choices=["mute", "hallucinate", "translate"],
        help="Objective of attack - hidden universal command/control.",
    )
    commandLineParser.add_argument(
        "--max_epochs", type=int, default=20, help="Training epochs for attack"
    )
    commandLineParser.add_argument(
        "--save_freq", type=int, default=1, help="Epoch frequency for saving attack"
    )
    commandLineParser.add_argument(
        "--attack_size", type=int, default=5120, help="Length of attack segment"
    )
    commandLineParser.add_argument(
        "--bs", type=int, default=16, help="Batch size for training attack"
    )
    commandLineParser.add_argument(
        "--lr", type=float, default=1e-3, help="Adversarial Attack learning rate"
    )
    commandLineParser.add_argument(
        "--clip_val",
        type=float,
        default=-1,
        help="Value (maximum) to clip the log mel vectors. -1 means no clipping",
    )
    commandLineParser.add_argument(
        "--attack_init",
        type=str,
        default="random",
        help="How to initialize attack. Give the path of a previously trained attack (model wrapper) if you want to initialize with it",
    )

    # eval attack args
    commandLineParser.add_argument(
        "--attack_epoch",
        type=int,
        default=-1,
        help="Specify which training epoch of attack to evaluate; -1 means no attack",
    )
    commandLineParser.add_argument(
        "--force_run",
        action="store_true",
        help="Do not load from cache",
    )
    commandLineParser.add_argument(
        "--not_none", action="store_true", help="Do not evaluate the none attack"
    )
    commandLineParser.add_argument(
        "--eval_train", action="store_true", help="Evaluate attack on the train split"
    )
    # commandLineParser.add_argument('--only_wer', action='store_true', help='Evaluate only the WER.')
    commandLineParser.add_argument(
        "--eval_metrics",
        type=str,
        default="nsl frac0",
        nargs="+",
        help="Which metrics to evaluate from: asl, frac0, wer, frac_lang",
    )
    commandLineParser.add_argument(
        "--frac_lang_langs",
        type=str,
        default="en fr",
        nargs="+",
        help="Which languages to evaluate for frac_lang metric",
    )

    # eval attack args for attack transferability
    commandLineParser.add_argument(
        "--transfer",
        action="store_true",
        help="Indicate it is a transferability attack (across model or dataset) for mel whitebox attack",
    )
    commandLineParser.add_argument(
        "--attack_model_dir",
        type=str,
        default="",
        help="path to trained attack to evaluate",
    )
    return commandLineParser.parse_known_args()


def analysis_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument(
        "--spectrogram", action="store_true", help="do analysis to generate spectrogram"
    )
    commandLineParser.add_argument(
        "--compare_with_audio", action="store_true", help="Include a real audio file"
    )
    commandLineParser.add_argument(
        "--sample_id",
        type=int,
        default=42,
        help="Specify which data sample to compare to",
    )

    commandLineParser.add_argument(
        "--wer_no_0", action="store_true", help="WER of non-zero length samples"
    )
    commandLineParser.add_argument(
        "--no_attack_path",
        type=str,
        default="",
        help="path to predictions with no attack",
    )
    commandLineParser.add_argument(
        "--attack_path", type=str, default="", help="path to predictions with attack"
    )

    commandLineParser.add_argument(
        "--saliency",
        action="store_true",
        help="Do saliency analysis. If you want to get saliency for a transfer attack - use attack transferability arguments and attack_path argument",
    )
    commandLineParser.add_argument(
        "--saliency_plot",
        action="store_true",
        help="Plot frame-level saliency across the audio recording.",
    )
    commandLineParser.add_argument(
        "--model_transfer_check", action="store_true", help="Determine if its possible for a muting attack is to transfer between different target models (passed in core_args.model_names)"
    )

    commandLineParser.add_argument(
        "--model_emb_close_exs", action="store_true", help="Print the 10 closest words for target tokens as per the embedding matrix.)"
    )


    return commandLineParser.parse_known_args()
