#!/usr/bin/env python3

import wave
from pathlib import Path

import numpy as np
from openwakeword.utils import AudioFeatures


ROOT = Path("/home/subash/thesis-social-robot")

OLD_DATA = (
    ROOT
    / "models/stop_robot/"
    / "training_data_v1/stop_robot"
)

IAEC_DIR = (
    ROOT
    / "models/stop_robot_iaec"
)

PLAYBACK_RECORDING = (
    IAEC_DIR
    / "data/playback_only/playback_002_false_trigger.wav"
)

OUTPUT_DIR = (
    IAEC_DIR
    / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

BATCH_SIZE = 64

# Additional examples containing only Piper.
PLAYBACK_ONLY_TRAIN = 500
PLAYBACK_ONLY_TEST = 250


def read_mono_wav(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()

        if channels != 1:
            raise ValueError(
                f"{path.name}: expected mono"
            )

        if rate != SAMPLE_RATE:
            raise ValueError(
                f"{path.name}: expected 16000 Hz"
            )

        if width != 2:
            raise ValueError(
                f"{path.name}: expected 16-bit"
            )

        audio = np.frombuffer(
            wav.readframes(
                wav.getnframes()
            ),
            dtype=np.int16,
        ).copy()

    return audio


def read_six_channel_wav(path):
    with wave.open(str(path), "rb") as wav:
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
                "Expected 16-bit recording"
            )

        audio = np.frombuffer(
            wav.readframes(
                wav.getnframes()
            ),
            dtype=np.int16,
        ).reshape(-1, channels)

    return audio


def rms(audio):
    return np.sqrt(
        np.mean(
            audio.astype(np.float64) ** 2
        )
        + 1e-12
    )


def make_echo_windows(
    six_channel_audio,
    start_times,
):
    windows = []

    for start_seconds in start_times:
        start = int(
            start_seconds * SAMPLE_RATE
        )

        end = start + WINDOW_SAMPLES

        if end > len(six_channel_audio):
            continue

        microphone_echo = (
            six_channel_audio[
                start:end,
                0,
            ].copy()
        )

        playback_reference = (
            six_channel_audio[
                start:end,
                5,
            ].copy()
        )

        windows.append(
            (
                microphone_echo,
                playback_reference,
            )
        )

    if not windows:
        raise RuntimeError(
            "No playback windows were created"
        )

    return windows


def mix_voice_with_echo(
    voice,
    microphone_echo,
    rng,
):
    voice_buffer = np.zeros(
        WINDOW_SAMPLES,
        dtype=np.float64,
    )

    voice_float = (
        voice.astype(np.float64)
        / 32768.0
    )

    if len(voice_float) > WINDOW_SAMPLES:
        voice_float = voice_float[
            :WINDOW_SAMPLES
        ]

    maximum_offset = (
        WINDOW_SAMPLES
        - len(voice_float)
    )

    offset = int(
        rng.integers(
            0,
            maximum_offset + 1,
        )
    )

    voice_buffer[
        offset:
        offset + len(voice_float)
    ] = voice_float

    echo_float = (
        microphone_echo.astype(np.float64)
        / 32768.0
    )

    echo_rms = rms(echo_float)
    voice_rms = rms(voice_buffer)

    # Train across difficult and easy overlap levels.
    target_sir_db = rng.uniform(
        -8.0,
        10.0,
    )

    target_voice_rms = (
        echo_rms
        * 10.0 ** (
            target_sir_db / 20.0
        )
    )

    voice_gain = (
        target_voice_rms
        / max(voice_rms, 1e-8)
    )

    mixed = (
        echo_float
        + voice_buffer * voice_gain
    )

    mixed = np.clip(
        mixed,
        -1.0,
        1.0,
    )

    return (
        mixed * 32767.0
    ).astype(np.int16)


six_channel_audio = read_six_channel_wav(
    PLAYBACK_RECORDING
)

# Different playback sections for training and testing.
train_start_times = np.arange(
    1.5,
    17.01,
    0.25,
)

test_start_times = np.arange(
    6.0,
    8.01,
    0.25,
)

train_echo_windows = make_echo_windows(
    six_channel_audio,
    train_start_times,
)

test_echo_windows = make_echo_windows(
    six_channel_audio,
    test_start_times,
)

print(
    "Training playback windows:",
    len(train_echo_windows),
)

print(
    "Testing playback windows:",
    len(test_echo_windows),
)


feature_extractor = AudioFeatures(
    sr=SAMPLE_RATE,
    ncpu=4,
    inference_framework="onnx",
    device="cpu",
)


def build_feature_pair(
    voice_paths,
    echo_windows,
    playback_only_count,
    seed,
    label,
):
    rng = np.random.default_rng(seed)

    items = list(voice_paths)

    # None means a playback-only negative.
    items.extend(
        [None] * playback_only_count
    )

    microphone_features = []
    reference_features = []

    total = len(items)

    for batch_start in range(
        0,
        total,
        BATCH_SIZE,
    ):
        batch_items = items[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        microphone_batch = []
        reference_batch = []

        for item in batch_items:
            window_index = int(
                rng.integers(
                    0,
                    len(echo_windows),
                )
            )

            (
                microphone_echo,
                playback_reference,
            ) = echo_windows[
                window_index
            ]

            if item is None:
                microphone_input = (
                    microphone_echo.copy()
                )

            else:
                voice = read_mono_wav(
                    item
                )

                microphone_input = (
                    mix_voice_with_echo(
                        voice,
                        microphone_echo,
                        rng,
                    )
                )

            microphone_batch.append(
                microphone_input
            )

            reference_batch.append(
                playback_reference
            )

        microphone_batch = np.stack(
            microphone_batch
        )

        reference_batch = np.stack(
            reference_batch
        )

        microphone_embedding = (
            feature_extractor.embed_clips(
                microphone_batch,
                batch_size=BATCH_SIZE,
                ncpu=4,
            ).astype(np.float32)
        )

        reference_embedding = (
            feature_extractor.embed_clips(
                reference_batch,
                batch_size=BATCH_SIZE,
                ncpu=4,
            ).astype(np.float32)
        )

        microphone_features.append(
            microphone_embedding
        )

        reference_features.append(
            reference_embedding
        )

        completed = min(
            batch_start + BATCH_SIZE,
            total,
        )

        print(
            label,
            completed,
            "/",
            total,
        )

    microphone_features = np.concatenate(
        microphone_features,
        axis=0,
    )

    reference_features = np.concatenate(
        reference_features,
        axis=0,
    )

    print(
        label,
        "microphone:",
        microphone_features.shape,
    )

    print(
        label,
        "reference:",
        reference_features.shape,
    )

    return (
        microphone_features,
        reference_features,
    )


positive_train_paths = sorted(
    (OLD_DATA / "positive_train")
    .glob("*.wav")
)

positive_test_paths = sorted(
    (OLD_DATA / "positive_test")
    .glob("*.wav")
)

negative_train_paths = sorted(
    (OLD_DATA / "negative_train")
    .glob("*.wav")
)

negative_test_paths = sorted(
    (OLD_DATA / "negative_test")
    .glob("*.wav")
)


datasets = [
    (
        "positive_train",
        positive_train_paths,
        train_echo_windows,
        0,
        101,
    ),
    (
        "positive_test",
        positive_test_paths,
        test_echo_windows,
        0,
        202,
    ),
    (
        "negative_train",
        negative_train_paths,
        train_echo_windows,
        PLAYBACK_ONLY_TRAIN,
        303,
    ),
    (
        "negative_test",
        negative_test_paths,
        test_echo_windows,
        PLAYBACK_ONLY_TEST,
        404,
    ),
]


for (
    name,
    paths,
    echo_windows,
    playback_only_count,
    seed,
) in datasets:

    microphone_features, reference_features = (
        build_feature_pair(
            paths,
            echo_windows,
            playback_only_count,
            seed,
            name,
        )
    )

    microphone_output = (
        OUTPUT_DIR
        / f"{name}_microphone.npy"
    )

    reference_output = (
        OUTPUT_DIR
        / f"{name}_reference.npy"
    )

    np.save(
        microphone_output,
        microphone_features,
    )

    np.save(
        reference_output,
        reference_features,
    )

    print(
        "Saved:",
        microphone_output,
    )

    print(
        "Saved:",
        reference_output,
    )


print(
    "Paired feature generation complete."
)
