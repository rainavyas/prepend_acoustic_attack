from .whisper import WhisperModel, WhisperModelEnsemble

def load_model(core_args, device=None):
    if len(core_args.model_name) > 1:
        return WhisperModelEnsemble(core_args.model_name, device=device, task=core_args.task, language=core_args.language)
    else:
        return WhisperModel(core_args.model_name[0], device=device, task=core_args.task, language=core_args.language)