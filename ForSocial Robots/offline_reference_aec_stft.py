#!/usr/bin/env python3

import wave
from pathlib import Path

import numpy as np
from scipy.signal import istft, stft


ROOT = Path("/home/subash/thesis-social-robot")

INPUT_FILE = (
    ROOT
    / "test_audio"
    / "reference_aec_new_6ch.wav"
)

CLEANED_FILE = (
    ROOT
    / "test_audio"
    / "reference_aec_stft_cleaned.wav"
)

ECHO_FILE = (
    ROOT
    / "test_audio"
    / "reference_aec_stft_estimated_echo.wav"
)

SAMPLE_RATE = 16000

# This interval contains Piper but not your command.
TRAIN_START_SECONDS = 2.2
TRAIN_END_SECONDS = 4.8

FFT_SIZE = 2048
HOP_SIZE = 512
OVERLAP = FFT_SIZE - HOP_SIZE

EPSILON = 1e-10


def save_mono(path, audio):
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    audio_int16 = np.clip(
        audio * 32768.0,
        -32768,
        32767,
    ).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(
            audio_int16.tobytes()
        )


with wave.open(str(INPUT_FILE), "rb") as wav:
    channels = wav.getnchannels()
    rate = wav.getframerate()
    width = wav.getsampwidth()

    if channels != 6:
        raise ValueError(
            f"Expected 6 channels, found {channels}"
        )

    if rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected 16000 Hz, found {rate}"
        )

    if width != 2:
        raise ValueError(
            f"Expected 16-bit audio, found {width * 8}-bit"
        )

    audio = np.frombuffer(
        wav.readframes(
            wav.getnframes()
        ),
        dtype=np.int16,
    ).reshape(-1, channels)


# Channel 0 contains your voice plus Piper.
microphone = (
    audio[:, 0].astype(np.float64)
    / 32768.0
)

# Channel 5 contains only the Piper reference.
reference = (
    audio[:, 5].astype(np.float64)
    / 32768.0
)


frequencies, times, microphone_stft = stft(
    microphone,
    fs=SAMPLE_RATE,
    nperseg=FFT_SIZE,
    noverlap=OVERLAP,
    boundary="zeros",
    padded=True,
)

_, _, reference_stft = stft(
    reference,
    fs=SAMPLE_RATE,
    nperseg=FFT_SIZE,
    noverlap=OVERLAP,
    boundary="zeros",
    padded=True,
)


training_frames = (
    (times >= TRAIN_START_SECONDS)
    & (times <= TRAIN_END_SECONDS)
)

if not np.any(training_frames):
    raise RuntimeError(
        "No STFT frames found in training interval"
    )


microphone_training = microphone_stft[
    :,
    training_frames,
]

reference_training = reference_stft[
    :,
    training_frames,
]


# Learn how the Piper reference appears in channel 0.
cross_power = np.sum(
    microphone_training
    * np.conj(reference_training),
    axis=1,
)

reference_power = np.sum(
    np.abs(reference_training) ** 2,
    axis=1,
)

regularization = (
    1e-4 * np.max(reference_power)
    + EPSILON
)

echo_path = (
    cross_power
    / (
        reference_power
        + regularization
    )
)


# Estimate and subtract Piper from channel 0.
estimated_echo_stft = (
    echo_path[:, None]
    * reference_stft
)

cleaned_stft = (
    microphone_stft
    - estimated_echo_stft
)


_, estimated_echo = istft(
    estimated_echo_stft,
    fs=SAMPLE_RATE,
    nperseg=FFT_SIZE,
    noverlap=OVERLAP,
    input_onesided=True,
    boundary=True,
)

_, cleaned = istft(
    cleaned_stft,
    fs=SAMPLE_RATE,
    nperseg=FFT_SIZE,
    noverlap=OVERLAP,
    input_onesided=True,
    boundary=True,
)


estimated_echo = estimated_echo[
    :len(microphone)
]

cleaned = cleaned[
    :len(microphone)
]


train_start = int(
    TRAIN_START_SECONDS * SAMPLE_RATE
)

train_end = int(
    TRAIN_END_SECONDS * SAMPLE_RATE
)

before_rms = np.sqrt(
    np.mean(
        microphone[
            train_start:train_end
        ] ** 2
    )
)

after_rms = np.sqrt(
    np.mean(
        cleaned[
            train_start:train_end
        ] ** 2
    )
)

change_db = 20.0 * np.log10(
    (
        after_rms + EPSILON
    )
    / (
        before_rms + EPSILON
    )
)


print(
    "Training STFT frames:",
    int(np.sum(training_frames)),
)

print(
    "Robot-only RMS before:",
    round(before_rms, 6),
)

print(
    "Robot-only RMS after:",
    round(after_rms, 6),
)

print(
    "Robot-only change:",
    round(change_db, 2),
    "dB",
)


save_mono(
    ECHO_FILE,
    estimated_echo,
)

save_mono(
    CLEANED_FILE,
    cleaned,
)

print(
    "Saved estimated echo:",
    ECHO_FILE,
)

print(
    "Saved cleaned audio:",
    CLEANED_FILE,
)
