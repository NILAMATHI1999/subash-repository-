#!/usr/bin/env python3

import wave
from pathlib import Path

import numpy as np
from openwakeword.utils import AudioFeatures


ROOT = Path("/home/subash/thesis-social-robot")

DATA_DIR = (
    ROOT
    / "models/stop_robot_iaec/data"
)

OUTPUT_DIR = (
    ROOT
    / "models/stop_robot_iaec/features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SAMPLE_RATE = 16000
WINDOW_SAMPLES = 2 * SAMPLE_RATE


POSITIVE_FILE = (
    DATA_DIR
    / "positive/subash_001.wav"
)

NEGATIVE_FILES = [
    DATA_DIR
    / "partial_stop/subash_stop_001.wav",

    DATA_DIR
    / "partial_robot/subash_robot_001.wav",

    DATA_DIR
    / "unrelated_speech/subash_unrelated_001.wav",
]

PLAYBACK_ONLY_FILE = (
    DATA_DIR
    / "playback_only/playback_001.wav"
)


def read_six_channel(path):
    with wave.open(str(path), "rb") as wav:

        if wav.getnchannels() != 6:
            raise ValueError(
                f"{path.name}: expected 6 channels"
            )

        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"{path.name}: expected 16000 Hz"
            )

        if wav.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: expected 16-bit"
            )

        audio = np.frombuffer(
            wav.readframes(
                wav.getnframes()
            ),
            dtype=np.int16,
        ).reshape(-1, 6)

    return audio


def extract_windows(
    path,
    start_times,
):
    audio = read_six_channel(path)

    microphone_windows = []
    reference_windows = []

    for start_seconds in start_times:

        start = int(
            start_seconds * SAMPLE_RATE
        )

        end = start + WINDOW_SAMPLES

        if end > len(audio):
            continue

        microphone_windows.append(
            audio[
                start:end,
                0,
            ].copy()
        )

        reference_windows.append(
            audio[
                start:end,
                5,
            ].copy()
        )

    print(
        path.name,
        "->",
        len(microphone_windows),
        "paired windows",
    )

    return (
        microphone_windows,
        reference_windows,
    )


# The command was spoken after the five-second prompt.
utterance_start_times = np.arange(
    4.4,
    5.21,
    0.08,
)

# Playback-only windows cover different Piper sentences.
playback_start_times = np.arange(
    2.0,
    8.01,
    0.25,
)


positive_microphone, positive_reference = (
    extract_windows(
        POSITIVE_FILE,
        utterance_start_times,
    )
)


negative_microphone = []
negative_reference = []

for negative_file in NEGATIVE_FILES:

    microphone_windows, reference_windows = (
        extract_windows(
            negative_file,
            utterance_start_times,
        )
    )

    negative_microphone.extend(
        microphone_windows
    )

    negative_reference.extend(
        reference_windows
    )


playback_microphone, playback_reference = (
    extract_windows(
        PLAYBACK_ONLY_FILE,
        playback_start_times,
    )
)

negative_microphone.extend(
    playback_microphone
)

negative_reference.extend(
    playback_reference
)


positive_microphone = np.stack(
    positive_microphone
)

positive_reference = np.stack(
    positive_reference
)

negative_microphone = np.stack(
    negative_microphone
)

negative_reference = np.stack(
    negative_reference
)


print(
    "Positive audio:",
    positive_microphone.shape,
)

print(
    "Negative audio:",
    negative_microphone.shape,
)


feature_extractor = AudioFeatures(
    sr=SAMPLE_RATE,
    ncpu=4,
    inference_framework="onnx",
    device="cpu",
)


positive_microphone_features = (
    feature_extractor.embed_clips(
        positive_microphone,
        batch_size=64,
        ncpu=4,
    ).astype(np.float32)
)

positive_reference_features = (
    feature_extractor.embed_clips(
        positive_reference,
        batch_size=64,
        ncpu=4,
    ).astype(np.float32)
)

negative_microphone_features = (
    feature_extractor.embed_clips(
        negative_microphone,
        batch_size=64,
        ncpu=4,
    ).astype(np.float32)
)

negative_reference_features = (
    feature_extractor.embed_clips(
        negative_reference,
        batch_size=64,
        ncpu=4,
    ).astype(np.float32)
)


outputs = {
    "real_positive_microphone.npy":
        positive_microphone_features,

    "real_positive_reference.npy":
        positive_reference_features,

    "real_negative_microphone.npy":
        negative_microphone_features,

    "real_negative_reference.npy":
        negative_reference_features,
}


for filename, array in outputs.items():

    output = (
        OUTPUT_DIR
        / filename
    )

    np.save(
        output,
        array,
    )

    print(
        filename,
        array.shape,
    )


print(
    "Real paired feature extraction complete."
)
