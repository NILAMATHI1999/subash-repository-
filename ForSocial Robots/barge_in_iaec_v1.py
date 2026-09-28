#!/usr/bin/env python3

import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from openwakeword.utils import AudioFeatures


ROOT = Path("/home/subash/thesis-social-robot")

MODEL_PATH = (
    ROOT
    / "models/stop_robot_iaec/"
    / "stop_robot_iaec_v1.onnx"
)

PLAYBACK_FILE = (
    ROOT
    / "test_audio/barge_in_long_reply.wav"
)

CAPTURE_DEVICE = "hw:1,0"

PLAYBACK_TARGET = (
    "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__"
    "UAC1.0_-00.analog-stereo"
)


SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

FRAME_SAMPLES = 1280

FRAME_BYTES = (
    FRAME_SAMPLES
    * CHANNELS
    * SAMPLE_WIDTH
)

WARMUP_FRAMES = 16
THRESHOLD = 0.5
PROMPT_AFTER_SECONDS = 3.0


def read_exact(stream, size):
    data = b""

    while len(data) < size:
        part = stream.read(
            size - len(data)
        )

        if not part:
            return None

        data += part

    return data


def stop_process(process):
    if process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=2)

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


print("Loading dual-input iAEC model...")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider",
    ],
)

input_names = [
    item.name
    for item in session.get_inputs()
]

print(
    "Model inputs:",
    input_names,
)


microphone_preprocessor = AudioFeatures(
    sr=SAMPLE_RATE,
    ncpu=1,
    inference_framework="onnx",
    device="cpu",
)

reference_preprocessor = AudioFeatures(
    sr=SAMPLE_RATE,
    ncpu=1,
    inference_framework="onnx",
    device="cpu",
)

print("Model and preprocessors ready.")


recorder = subprocess.Popen(
    [
        "arecord",
        "-D",
        CAPTURE_DEVICE,
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE),
        "-c",
        str(CHANNELS),
        "--buffer-time=1000000",
        "-t",
        "raw",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)

captured_frames = []


print("Warming synchronized audio buffers...")

for _ in range(WARMUP_FRAMES):

    raw = read_exact(
        recorder.stdout,
        FRAME_BYTES,
    )

    if raw is None:
        raise RuntimeError(
            "Microphone ended during warm-up"
        )

    captured_frames.append(raw)

    six_channel_frame = np.frombuffer(
        raw,
        dtype=np.int16,
    ).reshape(
        -1,
        CHANNELS,
    )

    microphone_frame = (
        six_channel_frame[
            :,
            0,
        ].copy()
    )

    reference_frame = (
        six_channel_frame[
            :,
            5,
        ].copy()
    )

    microphone_preprocessor(
        microphone_frame
    )

    reference_preprocessor(
        reference_frame
    )


print("Warm-up complete.")


playback = subprocess.Popen(
    [
        "pw-play",
        "--target",
        PLAYBACK_TARGET,
        str(PLAYBACK_FILE),
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

playback_started = time.perf_counter()

prompt_time = None
command_detected = False
false_stop_before_prompt = False

maximum_score = 0.0
maximum_score_time = 0.0


print("Playback started.")
print("Wait for the SAY NOW message.")


try:
    while playback.poll() is None:

        raw = read_exact(
            recorder.stdout,
            FRAME_BYTES,
        )

        if raw is None:
            print(
                "Microphone stream ended."
            )
            break

        captured_frames.append(raw)

        six_channel_frame = np.frombuffer(
            raw,
            dtype=np.int16,
        ).reshape(
            -1,
            CHANNELS,
        )

        microphone_frame = (
            six_channel_frame[
                :,
                0,
            ].copy()
        )

        reference_frame = (
            six_channel_frame[
                :,
                5,
            ].copy()
        )

        microphone_preprocessor(
            microphone_frame
        )

        reference_preprocessor(
            reference_frame
        )

        microphone_features = (
            microphone_preprocessor
            .get_features(
                n_feature_frames=16
            )
            .astype(np.float32)
        )

        reference_features = (
            reference_preprocessor
            .get_features(
                n_feature_frames=16
            )
            .astype(np.float32)
        )

        score = float(
            session.run(
                ["score"],
                {
                    "microphone_features":
                        microphone_features,

                    "reference_features":
                        reference_features,
                },
            )[0].reshape(-1)[0]
        )

        now = time.perf_counter()

        playback_elapsed = (
            now - playback_started
        )

        if score > maximum_score:
            maximum_score = score
            maximum_score_time = (
                playback_elapsed
            )

        if (
            prompt_time is None
            and playback_elapsed
            >= PROMPT_AFTER_SECONDS
        ):
            prompt_time = now

            print()

            print(
                ">>> SAY ONCE NOW: Stop robot",
                flush=True,
            )

        if score >= 0.10:
            print(
                f"Score: {score:.4f} "
                f"at {playback_elapsed:.2f} s"
            )

        if score < THRESHOLD:
            continue

        command_detected = True

        detection_time = (
            time.perf_counter()
        )

        print()

        if prompt_time is None:
            false_stop_before_prompt = True

            print(
                "FALSE STOP: iAEC triggered "
                "before the user prompt."
            )

        else:
            print(
                "STOP ROBOT detected by "
                "the dual-input model."
            )

            print(
                "Prompt-to-detection time:",
                round(
                    detection_time
                    - prompt_time,
                    2,
                ),
                "seconds",
            )

        stop_process(playback)

        print("Playback interrupted.")

        break


    if not command_detected:
        print(
            "STOP ROBOT was not detected "
            "before playback finished."
        )


finally:
    stop_process(playback)
    stop_process(recorder)


timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

capture_file = (
    ROOT
    / "test_audio"
    / f"barge_in_iaec_v1_{timestamp}.wav"
)


with wave.open(
    str(capture_file),
    "wb",
) as wav:

    wav.setnchannels(CHANNELS)
    wav.setsampwidth(SAMPLE_WIDTH)
    wav.setframerate(SAMPLE_RATE)

    wav.writeframes(
        b"".join(captured_frames)
    )


print(
    "Maximum score:",
    round(maximum_score, 4),
)

print(
    "Maximum-score time:",
    round(maximum_score_time, 2),
    "seconds",
)

print(
    "Threshold:",
    THRESHOLD,
)

print(
    "Result:",
    (
        "FALSE STOP"
        if false_stop_before_prompt
        else "SUCCESS"
        if command_detected
        else "NO DETECTION"
    ),
)

print(
    "Saved synchronized capture:",
    capture_file,
)
