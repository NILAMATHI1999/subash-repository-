#!/usr/bin/env python3

import re
import signal
import subprocess
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from openwakeword.model import Model


ROOT = Path("/home/subash/thesis-social-robot")

WAKEWORD_MODEL_PATH = (
    ROOT / "models/stop_robot/stop_robot_v6.onnx"
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

WAKEWORD_THRESHOLD = 0.5
WARMUP_FRAMES = 16
PROMPT_AFTER_SECONDS = 3.0

# Keep the newest 3.2 seconds for verification.
VERIFY_BUFFER_FRAMES = 24

# After a candidate, capture another 0.48 seconds.
TRAILING_FRAMES = 12

COMMAND_PATTERN = re.compile(
    r"\bstop[\s,;:!\-]+robots?\b",
    flags=re.IGNORECASE,
)


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


print("Loading openWakeWord V6...")

wakeword_model = Model(
    wakeword_models=[
        str(WAKEWORD_MODEL_PATH)
    ],
    inference_framework="onnx",
    enable_speex_noise_suppression=False,
)

model_key = next(
    iter(wakeword_model.models)
)

print("Loading Faster-Whisper base...")

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
)

print("Models ready.")


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
        "FR",
        "--format",
        "s16",
        "-",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)

captured_frames = []

verification_buffer = deque(
    maxlen=VERIFY_BUFFER_FRAMES
)


print("Warming microphone buffer...")

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

    frame = np.frombuffer(
        raw,
        dtype=np.int16,
    )

    wakeword_model.predict(frame)


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
command_verified = False
candidate_count = 0

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
        verification_buffer.append(raw)

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
            wakeword_model.predict(
                frame
            )[model_key]
        )

        if score < WAKEWORD_THRESHOLD:
            continue

        candidate_count += 1

        print()

        print(
            "Candidate",
            candidate_count,
            "score:",
            round(score, 4),
        )
        if playback.poll() is None:
            playback.send_signal(
                signal.SIGSTOP
            )

            print(
                "Playback temporarily paused."
            )

        # Capture the end of the phrase before verification.
        for _ in range(TRAILING_FRAMES):

            trailing_raw = read_exact(
                recorder.stdout,
                FRAME_BYTES,
            )

            if trailing_raw is None:
                break

            captured_frames.append(
                trailing_raw
            )

            verification_buffer.append(
                trailing_raw
            )

            trailing_frame = np.frombuffer(
                trailing_raw,
                dtype=np.int16,
            )

            wakeword_model.predict(
                trailing_frame
            )

        verification_audio = np.frombuffer(
            b"".join(
                verification_buffer
            ),
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        verification_started = (
            time.perf_counter()
        )

        segments, _ = whisper_model.transcribe(
            verification_audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        verification_text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        verification_time = (
            time.perf_counter()
            - verification_started
        )

        print(
            "Whisper verification:",
            repr(verification_text),
        )

        print(
            "Verification inference:",
            round(
                verification_time,
                2,
            ),
            "seconds",
        )

        if not COMMAND_PATTERN.search(
            verification_text
        ):
            print(
                "Candidate rejected: complete "
                "'stop robot' was not verified."
            )

            if playback.poll() is None:
                playback.send_signal(
                    signal.SIGCONT
                )

                print("Playback resumed.")

            continue

        command_verified = True

        detection_time = (
            time.perf_counter()
        )

        print(
            "COMPLETE COMMAND VERIFIED."
        )

        if prompt_time is not None:
            print(
                "Prompt-to-verification time:",
                round(
                    detection_time
                    - prompt_time,
                    2,
                ),
                "seconds",
            )
            if playback.poll() is None:
                playback.send_signal(
                    signal.SIGCONT
                )

        stop_process(playback)

        print("Playback interrupted.")

        break

    if not command_verified:
        print(
            "Complete 'stop robot' command "
            "was not verified."
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
    / f"barge_in_cascade_v1_{timestamp}.wav"
)

with wave.open(
    str(capture_file),
    "wb",
) as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(
        b"".join(captured_frames)
    )


print(
    "Candidate count:",
    candidate_count,
)

print(
    "Result:",
    (
        "SUCCESS"
        if command_verified
        else "NO VERIFIED COMMAND"
    ),
)

print(
    "Saved capture:",
    capture_file,
)
