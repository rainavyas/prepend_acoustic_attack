from .whisper import WhisperModel, WhisperModelEnsemble
from .canary import CanaryModel

def load_model(core_args, device=None):
    if len(core_args.model_name) > 1:
        return WhisperModelEnsemble(core_args.model_name, device=device, task=core_args.task, language=core_args.language)
    else:
        if 'canary' in core_args.model_name[0]:
            # return None
            return CanaryModel(device=device, task=core_args.task, language=core_args.language, pnc=True)
        else:
            return WhisperModel(core_args.model_name[0], device=device, task=core_args.task, language=core_args.language)