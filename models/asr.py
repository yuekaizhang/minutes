#!/usr/bin/env python3
#
# Copyright (c)  2023 by manyeyes

"""
This file demonstrates how to use sherpa-onnx Python API to transcribe
file(s) with a non-streaming model.

(1) For paraformer
    ./transcribe.py  \
      --tokens=/path/to/tokens.txt \
      --paraformer=/path/to/paraformer.onnx \
      --num-threads=2 \
      --decoding-method=greedy_search \
      --debug=false \
      --sample-rate=16000 \
      --feature-dim=80 \
      /path/to/0.wav \
      /path/to/1.wav
"""
import argparse
import time

from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx
import datetime
import soundfile

from .vad import OnnxWrapper, get_speech_timestamps

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the feature extractor. Must match the one expected by the model. Note: The input sound files can have a different sample rate from this argument.",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to decode. Each file must be of WAVE"
        "format with a single channel, and each sample has 16-bit, "
        "i.e., int16_t. "
        "The sample rate of the file can be arbitrary and does not need to "
        "be 16 kHz",
    )

    return parser.parse_args()

def process_time(milliseconds):
    delta = datetime.timedelta(milliseconds=milliseconds)
    time_str = str(delta)
    time_parts = time_str.split(".")[0].split(":")
    time_hms = "{:02d}:{:02d}:{:02d}".format(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]))
    # time_hms = "{:02d}:{:02d}:{:02d}:{:03d}".format(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]), int(str(milliseconds)[-3:]))
    return time_hms

def transcribe_with_vad(recognizer, vad_model, wav: np.ndarray, sample_rate: int = 16000):
    timestamps = get_speech_timestamps(wav, vad_model)
    results_list = []
    for idx, timestamp in enumerate(timestamps):
        streams = []
        samples = wav[timestamp['start']:timestamp['end']]
        s = recognizer.create_stream()
        s.accept_waveform(sample_rate, samples)
        streams.append(s)
        recognizer.decode_streams(streams)
        results = [s.result.text for s in streams]

        timestamp['s'] = results[0]
        timestamp['id'] = idx
        timestamp['start_time'] = process_time(int(timestamp['start']/sample_rate*1000))
        timestamp['end_time'] = process_time(int(timestamp['end']/sample_rate*1000))
        results_list.append(timestamp)
        print(timestamp)
    return results_list

def main():
    args = get_args()

    assert args.num_threads > 0, args.num_threads

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=args.paraformer,
        tokens=args.tokens,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=args.feature_dim,
        decoding_method=args.decoding_method,
        debug=args.debug,
    )


    print("Started!")
    vad_model = OnnxWrapper("/mnt/samsung-t7/yuekai/hackthon/silero-vad/files/silero_vad.onnx")
    wav, sample_rate = soundfile.read("/mnt/samsung-t7/yuekai/hackthon/paraformerX/chat.wav")

    results = transcribe(recognizer, vad_model, wav, sample_rate)

if __name__ == "__main__":
    main()
