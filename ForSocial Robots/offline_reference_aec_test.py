#!/usr/bin/env python3

import wave
from pathlib import Path

import numpy as np
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.signal import lfilter


ROOT = Path("/home/subash/thesis-social-robot")

INPUT_FILE = (
    ROOT
    / "test_audio"
    / "software_aec_input_6ch.wav"
)

OUTPUT_FILE = (
    ROOT
    / "test_audio"
    / "software_aec_reference_cleaned.wav"
)

ECHO_FILE = (
    ROOT
    / "test_audio"
    / "software_aec_estimated_echo.wav"
)

MIC_FILE = (
    ROOT
    / "test_audio"
    / "software_aec_original_mic.wav"
)

SAMPLE_RATE = 16000

# Robot-only portion of the controlled recording.
TRAIN_START_SECONDS = 3.2
TRAIN_END_SECONDS = 4.8

# Maximum delay searched between reference and microphone.
MAX_LAG_SECONDS = 0.20

# Adaptive echo-path length: 16 milliseconds.
FILTER_LENGTH = 256

TRAINING_EPOCHS = 6
NLMS_STEP = 0.15
EPSILON = 1e-8


def save_mono(path, audio):
    audio_int16 = np.clip(
        audio * 32768.0,
        -32768,
        32767,
    ).astype(np.int16)

    with wave.open(
        str(path),
        "wb",
    ) as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(
            audio_int16.tobytes()
        )


def shift_reference(reference, lag):
    shifted = np.zeros_like(reference)

    if lag > 0:
        shifted[lag:] = reference[:-lag]

    elif lag < 0:
        shifted[:lag] = reference[-lag:]

    else:
        shifted[:] = reference

    return shifted


with wave.open(
    str(INPUT_FILE),
    "rb",
) as wav:

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


microphone = (
    audio[:, 0].astype(np.float64)
    / 32768.0
)

reference = (
    audio[:, 5].astype(np.float64)
    / 32768.0
)

train_start = int(
    TRAIN_START_SECONDS * SAMPLE_RATE
)

train_end = int(
    TRAIN_END_SECONDS * SAMPLE_RATE
)

microphone_train = microphone[
    train_start:train_end
]

reference_train = reference[
    train_start:train_end
]

max_lag = int(
    MAX_LAG_SECONDS * SAMPLE_RATE
)

cross_correlation = correlate(
    microphone_train,
    reference_train,
    mode="full",
    method="fft",
)

lags = correlation_lags(
    len(microphone_train),
    len(reference_train),
    mode="full",
)

valid = (
    (lags >= -max_lag)
    & (lags <= max_lag)
)

best_index = np.argmax(
    np.abs(
        cross_correlation[valid]
    )
)

best_lag = lags[valid][best_index]

aligned_reference = shift_reference(
    reference,
    int(best_lag),
)

print(
    "Estimated reference lag:",
    int(best_lag),
    "samples",
)

print(
    "Estimated reference lag:",
    round(
        1000.0
        * best_lag
        / SAMPLE_RATE,
        2,
    ),
    "milliseconds",
)


weights = np.zeros(
    FILTER_LENGTH,
    dtype=np.float64,
)

for epoch in range(
    1,
    TRAINING_EPOCHS + 1,
):

    squared_error = 0.0
    sample_count = 0

    for index in range(
        train_start + FILTER_LENGTH - 1,
        train_end,
    ):

        reference_window = aligned_reference[
            index - FILTER_LENGTH + 1:
            index + 1
        ][::-1]

        estimated_echo = np.dot(
            weights,
            reference_window,
        )

        error = (
            microphone[index]
            - estimated_echo
        )

        reference_power = np.dot(
            reference_window,
            reference_window,
        )

        weights += (
            NLMS_STEP
            * error
            * reference_window
            / (
                reference_power
                + EPSILON
            )
        )

        squared_error += error * error
        sample_count += 1

    epoch_rms = np.sqrt(
        squared_error
        / max(sample_count, 1)
    )

    print(
        "Training epoch",
        epoch,
        "error RMS:",
        round(epoch_rms, 6),
    )


estimated_echo = lfilter(
    weights,
    [1.0],
    aligned_reference,
)

cleaned = (
    microphone
    - estimated_echo
)


before_rms = np.sqrt(
    np.mean(
        microphone_train ** 2
    )
)

cleaned_train = cleaned[
    train_start:train_end
]

after_rms = np.sqrt(
    np.mean(
        cleaned_train ** 2
    )
)

attenuation_db = 20.0 * np.log10(
    (
        after_rms + EPSILON
    )
    / (
        before_rms + EPSILON
    )
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
    "Change:",
    round(attenuation_db, 2),
    "dB",
)


save_mono(
    MIC_FILE,
    microphone,
)

save_mono(
    ECHO_FILE,
    estimated_echo,
)

save_mono(
    OUTPUT_FILE,
    cleaned,
)

print(
    "Saved original microphone:",
    MIC_FILE,
)

print(
    "Saved estimated echo:",
    ECHO_FILE,
)

print(
    "Saved cleaned audio:",
    OUTPUT_FILE,
)
