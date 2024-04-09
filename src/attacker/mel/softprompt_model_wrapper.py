import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from tqdm import tqdm
import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import tqdm
from math import log

from src.tools.tools import set_seeds, AverageMeter
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.timing import add_word_timestamps
from whisper.tokenizer import get_tokenizer
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.utils import (
    exact_div,
    format_timestamp,
    # get_end,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)


class SoftPromptModelWrapper(nn.Module):
    '''
        Model with the only learnable parameters being the soft prompt vectors in the mel-space
    '''
    def __init__(self, tokenizer, num_dim=80, num_vectors=32, device=None):
        super(SoftPromptModelWrapper, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.softprompt = nn.Parameter(torch.rand(num_dim, num_vectors))

        self.num_vectors = num_vectors

    def forward(self, mels, whisper_model):
        '''
            mels: Torch.tensor: [Batch x 80 x N_FRAMES]
            whisper_model: encoder-decoder model

            Returns the logits for the first transcribed token
        '''

        # prepend softprompt
        mels = mels[:,:,:-self.num_vectors]
        X = self.softprompt.unsqueeze(0).expand(mels.size(0), -1, -1)
        sp_mels = torch.cat((X, mels), dim=2)

        return self._mel_to_logit(sp_mels, whisper_model)

    def _mel_to_logit(self, mel: torch.Tensor, whisper_model):
        '''
            Forward pass through the whisper model of the mel vectors
            expect mel vectors passed as a batch and padded to 30s of audio length
            mel: torch.Tensor [B x dim x num_vectors]
        '''
        # create batch of start of transcript tokens
        sot_ids = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps)
        sot_ids = sot_ids.to(self.device)
        sot_ids = sot_ids.unsqueeze(0).expand(mel.size(0), -1)
        return whisper_model.model.forward(mel, sot_ids)
    

    def transcribe(self,
        whisper_model,
        audio: Union[str, np.ndarray, torch.Tensor],
        *,
        do_mel_attack = True,
        k_scale = 1, # scale the mel-vector down by k to make it more imperceptible
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        
        **decode_options,
    ):
        '''
            Mimics the original Whisper transcribe functions but prepends the adversarial softprompt vectors
            in the mel-vector space

                do_mel_attack parameter is a boolean to do the softprompt attack or not

            For the original function refer to:
                 https://github.com/openai/whisper/blob/main/whisper/transcribe.py
        '''

        dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
        if dtype == torch.float32:
            decode_options["fp16"] = False
        decode_options["language"] = whisper_model.language
        
        mel = log_mel_spectrogram(audio, whisper_model.model.dims.n_mels, padding=N_SAMPLES).to(self.device)
        # Modify content frames to account for softprompt vectors
        if do_mel_attack:
            content_frames = mel.shape[-1] - N_FRAMES + self.num_vectors
        else:
            content_frames = mel.shape[-1] - N_FRAMES
        content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

        if do_mel_attack:
            # prepend softprompt adversarial vectors (scaled by k_scale)
            mel = mel[:,:-self.num_vectors]
            mel = torch.cat((self.softprompt-log(k_scale), mel), dim=1)

        language = whisper_model.language
        task = whisper_model.task

        if isinstance(clip_timestamps, str):
            clip_timestamps = [
                float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
            ]
        seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))

        punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

        if word_timestamps and task == "translate":
            warnings.warn("Word-level timestamps on translations may not be reliable.")

        def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
            temperatures = (
                [temperature] if isinstance(temperature, (int, float)) else temperature
            )
            decode_result = None

            for t in temperatures:
                kwargs = {**decode_options}
                if t > 0:
                    # disable beam_size and patience when t > 0
                    kwargs.pop("beam_size", None)
                    kwargs.pop("patience", None)
                else:
                    # disable best_of when t == 0
                    kwargs.pop("best_of", None)

                options = DecodingOptions(**kwargs, temperature=t)
                decode_result = whisper_model.model.decode(segment, options)

                needs_fallback = False
                if (
                    compression_ratio_threshold is not None
                    and decode_result.compression_ratio > compression_ratio_threshold
                ):
                    needs_fallback = True  # too repetitive
                if (
                    logprob_threshold is not None
                    and decode_result.avg_logprob < logprob_threshold
                ):
                    needs_fallback = True  # average log probability is too low
                if (
                    no_speech_threshold is not None
                    and decode_result.no_speech_prob > no_speech_threshold
                ):
                    needs_fallback = False  # silence
                if not needs_fallback:
                    break

            return decode_result

        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        input_stride = exact_div(
            N_FRAMES, whisper_model.model.dims.n_audio_ctx
        )  # mel frames per output token: 2
        time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)
        all_tokens = []
        all_segments = []
        prompt_reset_since = 0

        if initial_prompt is not None:
            initial_prompt_tokens = self.tokenizer.encode(" " + initial_prompt.strip())
            all_tokens.extend(initial_prompt_tokens)
        else:
            initial_prompt_tokens = []

        def new_segment(
            *, start: float, end: float, tokens: torch.Tensor, result: DecodingResult
        ):
            tokens = tokens.tolist()
            text_tokens = [token for token in tokens if token < self.tokenizer.eot]
            return {
                "seek": seek,
                "start": start,
                "end": end,
                "text": self.tokenizer.decode(text_tokens),
                "tokens": tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }

        # show the progress bar when verbose is False (if True, transcribed text will be printed)
        with tqdm.tqdm(
            total=content_frames, unit="frames", disable=verbose is not False
        ) as pbar:
            last_speech_timestamp = 0.0
            # NOTE: This loop is obscurely flattened to make the diff readable.
            # A later commit should turn this into a simpler nested loop.
            # for seek_clip_start, seek_clip_end in seek_clips:
            #     while seek < seek_clip_end
            while clip_idx < len(seek_clips):
                seek_clip_start, seek_clip_end = seek_clips[clip_idx]
                if seek < seek_clip_start:
                    seek = seek_clip_start
                if seek >= seek_clip_end:
                    clip_idx += 1
                    if clip_idx < len(seek_clips):
                        seek = seek_clips[clip_idx][0]
                    continue
                time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)
                segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)
                mel_segment = mel[:, seek : seek + segment_size]
                segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
                mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(self.device).to(dtype)

                decode_options["prompt"] = all_tokens[prompt_reset_since:]
                result: DecodingResult = decode_with_fallback(mel_segment)
                tokens = torch.tensor(result.tokens)

                if no_speech_threshold is not None:
                    # no voice activity check
                    should_skip = result.no_speech_prob > no_speech_threshold
                    if (
                        logprob_threshold is not None
                        and result.avg_logprob > logprob_threshold
                    ):
                        # don't skip if the logprob is high enough, despite the no_speech_prob
                        should_skip = False

                    if should_skip:
                        seek += segment_size  # fast-forward to the next segment boundary
                        continue

                previous_seek = seek
                current_segments = []

                # anomalous words are very long/short/improbable
                def word_anomaly_score(word: dict) -> float:
                    probability = word.get("probability", 0.0)
                    duration = word["end"] - word["start"]
                    score = 0.0
                    if probability < 0.15:
                        score += 1.0
                    if duration < 0.133:
                        score += (0.133 - duration) * 15
                    if duration > 2.0:
                        score += duration - 2.0
                    return score

                def is_segment_anomaly(segment: Optional[dict]) -> bool:
                    if segment is None or not segment["words"]:
                        return False
                    words = [w for w in segment["words"] if w["word"] not in punctuation]
                    words = words[:8]
                    score = sum(word_anomaly_score(w) for w in words)
                    return score >= 3 or score + 0.01 >= len(words)

                def next_words_segment(segments: List[dict]) -> Optional[dict]:
                    return next((s for s in segments if s["words"]), None)

                timestamp_tokens: torch.Tensor = tokens.ge(self.tokenizer.timestamp_begin)
                single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

                consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
                consecutive.add_(1)
                if len(consecutive) > 0:
                    # if the output contains two consecutive timestamp tokens
                    slices = consecutive.tolist()
                    if single_timestamp_ending:
                        slices.append(len(tokens))

                    last_slice = 0
                    for current_slice in slices:
                        sliced_tokens = tokens[last_slice:current_slice]
                        start_timestamp_pos = (
                            sliced_tokens[0].item() - self.tokenizer.timestamp_begin
                        )
                        end_timestamp_pos = (
                            sliced_tokens[-1].item() - self.tokenizer.timestamp_begin
                        )
                        current_segments.append(
                            new_segment(
                                start=time_offset + start_timestamp_pos * time_precision,
                                end=time_offset + end_timestamp_pos * time_precision,
                                tokens=sliced_tokens,
                                result=result,
                            )
                        )
                        last_slice = current_slice

                    if single_timestamp_ending:
                        # single timestamp at the end means no speech after the last timestamp.
                        seek += segment_size
                    else:
                        # otherwise, ignore the unfinished segment and seek to the last timestamp
                        last_timestamp_pos = (
                            tokens[last_slice - 1].item() - self.tokenizer.timestamp_begin
                        )
                        seek += last_timestamp_pos * input_stride
                else:
                    duration = segment_duration
                    timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                    if (
                        len(timestamps) > 0
                        and timestamps[-1].item() != self.tokenizer.timestamp_begin
                    ):
                        # no consecutive timestamps but it has a timestamp; use the last one.
                        last_timestamp_pos = (
                            timestamps[-1].item() - self.tokenizer.timestamp_begin
                        )
                        duration = last_timestamp_pos * time_precision

                    current_segments.append(
                        new_segment(
                            start=time_offset,
                            end=time_offset + duration,
                            tokens=tokens,
                            result=result,
                        )
                    )
                    seek += segment_size

                if word_timestamps:
                    add_word_timestamps(
                        segments=current_segments,
                        model=whisper_model.model,
                        tokenizer=self.tokenizer,
                        mel=mel_segment,
                        num_frames=segment_size,
                        prepend_punctuations=prepend_punctuations,
                        append_punctuations=append_punctuations,
                        last_speech_timestamp=last_speech_timestamp,
                    )

                    if not single_timestamp_ending:
                        last_word_end = get_end(current_segments)
                        if last_word_end is not None and last_word_end > time_offset:
                            seek = round(last_word_end * FRAMES_PER_SECOND)

                    # skip silence before possible hallucinations
                    if hallucination_silence_threshold is not None:
                        threshold = hallucination_silence_threshold
                        if not single_timestamp_ending:
                            last_word_end = get_end(current_segments)
                            if last_word_end is not None and last_word_end > time_offset:
                                remaining_duration = window_end_time - last_word_end
                                if remaining_duration > threshold:
                                    seek = round(last_word_end * FRAMES_PER_SECOND)
                                else:
                                    seek = previous_seek + segment_size

                        # if first segment might be a hallucination, skip leading silence
                        first_segment = next_words_segment(current_segments)
                        if first_segment is not None and is_segment_anomaly(first_segment):
                            gap = first_segment["start"] - time_offset
                            if gap > threshold:
                                seek = previous_seek + round(gap * FRAMES_PER_SECOND)
                                continue

                        # skip silence before any possible hallucination that is surrounded
                        # by silence or more hallucinations
                        hal_last_end = last_speech_timestamp
                        for si in range(len(current_segments)):
                            segment = current_segments[si]
                            if not segment["words"]:
                                continue
                            if is_segment_anomaly(segment):
                                next_segment = next_words_segment(
                                    current_segments[si + 1 :]
                                )
                                if next_segment is not None:
                                    hal_next_start = next_segment["words"][0]["start"]
                                else:
                                    hal_next_start = time_offset + segment_duration
                                silence_before = (
                                    segment["start"] - hal_last_end > threshold
                                    or segment["start"] < threshold
                                    or segment["start"] - time_offset < 2.0
                                )
                                silence_after = (
                                    hal_next_start - segment["end"] > threshold
                                    or is_segment_anomaly(next_segment)
                                    or window_end_time - segment["end"] < 2.0
                                )
                                if silence_before and silence_after:
                                    seek = round(
                                        max(time_offset + 1, segment["start"])
                                        * FRAMES_PER_SECOND
                                    )
                                    if content_duration - segment["end"] < threshold:
                                        seek = content_frames
                                    current_segments[si:] = []
                                    break
                            hal_last_end = segment["end"]

                    last_word_end = get_end(current_segments)
                    if last_word_end is not None:
                        last_speech_timestamp = last_word_end

                if verbose:
                    for segment in current_segments:
                        start, end, text = segment["start"], segment["end"], segment["text"]
                        line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                        print(make_safe(line))

                # if a segment is instantaneous or does not contain text, clear it
                for i, segment in enumerate(current_segments):
                    if segment["start"] == segment["end"] or segment["text"].strip() == "":
                        segment["text"] = ""
                        segment["tokens"] = []
                        segment["words"] = []

                all_segments.extend(
                    [
                        {"id": i, **segment}
                        for i, segment in enumerate(
                            current_segments, start=len(all_segments)
                        )
                    ]
                )
                all_tokens.extend(
                    [token for segment in current_segments for token in segment["tokens"]]
                )

                if not condition_on_previous_text or result.temperature > 0.5:
                    # do not feed the prompt tokens if a high temperature was used
                    prompt_reset_since = len(all_tokens)

                # update progress bar
                pbar.update(min(content_frames, seek) - previous_seek)

        result =  dict(
            text=self.tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
            segments=all_segments,
            language=language,
        )
        segments = []
        for segment in result['segments']:
            segments.append(segment['text'].strip())
        return ' '.join(segments)
