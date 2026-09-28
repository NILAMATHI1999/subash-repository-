#!/usr/bin/env python3

import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from openwakeword.model import Model


ROOT = Path("/home/subash/thesis-social-robot")

MODEL_PATH = (
    ROOT / "models/stop_robot/stop_robot_v4.onnx"
)

PLAYBACK_FILE = (
    ROOT / "test_audio/barge_in_long_reply.wav"
)

CAPTURE_SOURCE = (
    "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__"
    "UAC1.0_-00.analog-surround-21"
)

PLAYBACK_TARGET = (
    "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__"
    "UAC1.0_-00.analog-stereo"
)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280
FRAME_BYTES = FRAME_SAMPLES * 2

THRESHOLD = 0.5
PROMPT_AFTER_SECONDS = 3.0
WARMUP_FRAMES = 16


def read_exact(stream, size):
    data = b""

    while len(data) < size:
        part = stream.read(size - len(data))

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


print("Loading stop_robot_v4...")

model = Model(
    wakeword_models=[str(MODEL_PATH)],
    inference_framework="onnx",
    enable_speex_noise_suppression=False,
)

model_key = next(iter(model.models))

print("Model ready.")


recorder = subprocess.Popen(
    [
        "pw-cat",
        "--record",
        "--target",
        CAPTURE_SOURCE,
        "--rate",
        str(SAMPLE_RATE),
        "--channels",
        "1",
        "--channel-map",
        "FL",
        "--format",
        "s16",
        "-",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)

captured_frames = []

print("Warming microphone buffer...")

for _ in range(WARMUP_FRAMES):
    raw = read_exact(
        recorder.stdout,
        FRAME_BYTES,
    )

    if raw is None:
        raise RuntimeError(
            "Microphone stream ended during warm-up"
        )

    captured_frames.append(raw)

    model.predict(
        np.frombuffer(raw, dtype=np.int16)
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
stop_detected = False
false_stop_before_prompt = False

print("Playback started.")
print("Wait for the SAY NOW message.")


try:
    while playback.poll() is None:

        raw = read_exact(
            recorder.stdout,
            FRAME_BYTES,
        )

        if raw is None:
            print("Microphone stream ended.")
            break

        captured_frames.append(raw)

        now = time.perf_counter()

        playback_elapsed = (
            now - playback_started
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

        frame = np.frombuffer(
            raw,
            dtype=np.int16,
        )

        score = float(
            model.predict(frame)[model_key]
        )

        if score >= 0.10:
            print(
                f"Score: {score:.4f} "
                f"at {playback_elapsed:.2f} s"
            )

        if score >= THRESHOLD:

            stop_detected = True
            detection_time = time.perf_counter()

            if prompt_time is None:

                false_stop_before_prompt = True

                print()

                print(
                    "FALSE STOP: detector triggered "
                    "before the user prompt."
                )

            else:

                print()

                print(
                    "STOP ROBOT detected."
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

    if not stop_detected:
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
    / f"barge_in_openwakeword_v4_{timestamp}.wav"
)

with wave.open(str(capture_file), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(
        b"".join(captured_frames)
    )


print("Threshold:", THRESHOLD)

print(
    "Result:",
    (
        "FALSE STOP"
        if false_stop_before_prompt
        else "SUCCESS"
        if stop_detected
        else "FAILURE"
    ),
)

print(
    "Saved capture:",
    capture_file,
)
