#!/usr/bin/env python3

import wave
from pathlib import Path

import numpy as np
from openwakeword.model import Model


ROOT = Path("/home/subash/thesis-social-robot")

MODEL_PATH = (
    ROOT / "models/stop_robot/stop_robot_v5.onnx"
)

OUTPUT_PATH = (
    ROOT
    / "models/stop_robot/"
    "v5_false_trigger_negative_features.npy"
)

AUDIO_FILES = [
    ROOT
    / "test_audio/"
    "barge_in_openwakeword_v5_fr_20260927_205333.wav",

    ROOT
    / "test_audio/"
    "barge_in_openwakeword_v5_fr_20260927_205349.wav",

    ROOT
    / "test_audio/"
    "barge_in_openwakeword_v5_fr_20260927_205403.wav",
]

FRAME_SAMPLES = 1280
WARMUP_FRAMES = 16

features = []

for audio_path in AUDIO_FILES:

    model = Model(
        wakeword_models=[str(MODEL_PATH)],
        inference_framework="onnx",
        enable_speex_noise_suppression=False,
    )

    with wave.open(str(audio_path), "rb") as wav:

        if (
            wav.getnchannels() != 1
            or wav.getframerate() != 16000
        ):
            raise ValueError(
                f"Unexpected audio format: "
                f"{audio_path.name}"
            )

        audio = np.frombuffer(
            wav.readframes(
                wav.getnframes()
            ),
            dtype=np.int16,
        )

    count = 0

    for frame_number, start in enumerate(
        range(
            0,
            len(audio),
            FRAME_SAMPLES,
        )
    ):

        frame = audio[
            start:start + FRAME_SAMPLES
        ]

        if len(frame) < FRAME_SAMPLES:
            break

        model.predict(frame)

        if frame_number >= WARMUP_FRAMES:

            feature_window = (
                model.preprocessor
                .get_features(
                    n_feature_frames=16
                )[0]
                .astype(np.float32)
            )

            features.append(
                feature_window
            )

            count += 1

    print(
        audio_path.name,
        "->",
        count,
        "negative windows",
    )

if not features:
    raise RuntimeError(
        "No negative features were extracted"
    )

feature_array = np.stack(
    features
)

np.save(
    OUTPUT_PATH,
    feature_array,
)

print(
    "Features:",
    feature_array.shape,
)

print(
    "Saved:",
    OUTPUT_PATH,
)
