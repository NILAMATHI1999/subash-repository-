#!/usr/bin/env python3

import queue
import re
import subprocess
import threading
import time
import wave

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


# ============================================================
# Paths and audio devices
# ============================================================

ROOT = Path("/home/subash/thesis-social-robot")

PLAYBACK_FILE = (
    ROOT
    / "test_audio"
    / "barge_in_long_reply.wav"
)

CAPTURE_DEVICE = "hw:1,0"

PLAYBACK_TARGET = "respeaker_aec_sink"


# ============================================================
# ReSpeaker recording configuration
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

CHUNK_SECONDS = 0.1

CHUNK_FRAMES = int(
    SAMPLE_RATE * CHUNK_SECONDS
)

CHUNK_BYTES = (
    CHUNK_FRAMES
    * CHANNELS
    * SAMPLE_WIDTH
)


# ============================================================
# Whisper rolling-window configuration
# ============================================================

WINDOW_SECONDS = 1.0

WINDOW_CHUNKS = int(
    WINDOW_SECONDS
    / CHUNK_SECONDS
)

WINDOW_FRAMES = int(
    SAMPLE_RATE
    * WINDOW_SECONDS
)

HOP_SECONDS = 0.25

HOP_FRAMES = int(
    SAMPLE_RATE
    * HOP_SECONDS
)

QUEUE_SIZE = 2


# ============================================================
# Controlled command test
# ============================================================

SAY_STOP_AFTER_SECONDS = 3.0

COMMAND_TEST_SECONDS = 5.0


# ============================================================
# Shared state
# ============================================================

window_queue = queue.Queue(
    maxsize=QUEUE_SIZE
)

capture_chunks = []

capture_stop_event = threading.Event()

prompt_event = threading.Event()

test_finished_event = threading.Event()

test_failed_event = threading.Event()

dropped_windows = 0

prompt_time = None


# ============================================================
# Load Faster-Whisper
# ============================================================

print("Loading Faster-Whisper...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
)

print("Model ready.")


# ============================================================
# Microphone recording thread
# ============================================================

def capture_audio():
    global dropped_windows

    recent_chunks = deque(
        maxlen=WINDOW_CHUNKS
    )

    # Important:
    # this prevents several windows from being generated
    # immediately when the first 1-second buffer becomes ready.
    frames_since_window = -(
        WINDOW_FRAMES - HOP_FRAMES
    )

    arecord = subprocess.Popen(
        [
            "pw-cat",
            "--record",
            "--target",
            "respeaker_aec_source",
            "--rate",
            str(SAMPLE_RATE),
            "--channels",
            str(CHANNELS),
            "--channel-map",
            "FL",
            "--format",
            "s16",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    try:
        while not capture_stop_event.is_set():

            raw = arecord.stdout.read(
                CHUNK_BYTES
            )

            if not raw:
                break

            samples = np.frombuffer(
                raw,
                dtype=np.int16,
            )

            expected_samples = (
                CHUNK_FRAMES
                * CHANNELS
            )

            if len(samples) != expected_samples:
                continue

            samples = samples.reshape(
                -1,
                CHANNELS,
            )

            # ReSpeaker processed microphone channel
            channel_0 = samples[:, 0].copy()

            capture_chunks.append(
                channel_0
            )

            recent_chunks.append(
                channel_0
            )

            frames_since_window += len(
                channel_0
            )

            if (
                len(recent_chunks)
                == WINDOW_CHUNKS
                and frames_since_window
                >= HOP_FRAMES
            ):

                window = np.concatenate(
                    list(recent_chunks)
                )

                window = window[
                    -WINDOW_FRAMES:
                ]

                frames_since_window -= (
                    HOP_FRAMES
                )

                if window_queue.full():

                    try:
                        window_queue.get_nowait()

                        dropped_windows += 1

                    except queue.Empty:
                        pass

                try:
                    window_queue.put_nowait(
                        window
                    )

                except queue.Full:
                    dropped_windows += 1

    finally:

        if arecord.poll() is None:
            arecord.terminate()

        try:
            arecord.wait(
                timeout=2
            )

        except subprocess.TimeoutExpired:
            arecord.kill()


# ============================================================
# Start continuous microphone capture
# ============================================================

capture_thread = threading.Thread(
    target=capture_audio,
    daemon=True,
)

capture_thread.start()


# Give arecord a short moment to start
time.sleep(0.5)


# ============================================================
# Start Piper playback
# ============================================================

playback = subprocess.Popen(
    [
        "pw-play",
        "--target",
        PLAYBACK_TARGET,
        str(PLAYBACK_FILE),
    ]
)

playback_start = time.perf_counter()

print("Playback started.")
print("Wait for the SAY NOW message.")


# ============================================================
# SAY NOW prompt
# ============================================================

def prompt_user():
    global prompt_time

    time.sleep(
        SAY_STOP_AFTER_SECONDS
    )

    prompt_time = time.perf_counter()

    print()
    print(
        ">>> SAY NOW: Stop robot",
        flush=True,
    )

    # Detection becomes valid only now
    prompt_event.set()


prompt_thread = threading.Thread(
    target=prompt_user,
    daemon=True,
)

prompt_thread.start()


# ============================================================
# Automatic 5-second test deadline
# ============================================================

def end_test_on_timeout():

    # Wait until SAY NOW actually happens
    prompt_event.wait()

    finished_in_time = (
        test_finished_event.wait(
            COMMAND_TEST_SECONDS
        )
    )

    if finished_in_time:
        return

    test_failed_event.set()

    print()
    print(
        "TEST FAILURE: complete "
        "'stop robot' command was "
        "not detected within "
        f"{COMMAND_TEST_SECONDS:.0f} seconds."
    )

    if playback.poll() is None:
        playback.terminate()


timeout_thread = threading.Thread(
    target=end_test_on_timeout,
    daemon=True,
)

timeout_thread.start()


# ============================================================
# Faster-Whisper barge-in loop
# ============================================================

stop_detected = False

detection_time = None

try:

    while playback.poll() is None:

        try:
            window = window_queue.get(
                timeout=0.1
            )

        except queue.Empty:
            continue

        audio = (
            window.astype(np.float32)
            / 32768.0
        )

        inference_started = (
            time.perf_counter()
        )

        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        inference_time = (
            time.perf_counter()
            - inference_started
        )

        if text:

            print(
                f"Heard: {text} "
                f"| inference: "
                f"{inference_time:.2f} s"
            )

        # Ignore everything before SAY NOW
        if not prompt_event.is_set():
            continue

        # Require the complete command "stop robot"
        if re.search(
            r"\bstop\b[\s,;:!\-]+\brobot\b",
            text,
            flags=re.IGNORECASE,
        ):

            candidate_time = (
                time.perf_counter()
            )

            # Reject late detections
            if (
                test_failed_event.is_set()
                or prompt_time is None
                or candidate_time - prompt_time
                > COMMAND_TEST_SECONDS
            ):

                test_failed_event.set()

                if playback.poll() is None:
                    playback.terminate()

                break

            stop_detected = True

            detection_time = (
                candidate_time
            )

            test_finished_event.set()

            playback_elapsed = (
                detection_time
                - playback_start
            )

            prompt_elapsed = (
                detection_time
                - prompt_time
            )

            print()
            print(
                "TEST SUCCESS: complete "
                "'stop robot' command detected."
            )

            print(
                "STOP detected after "
                f"{playback_elapsed:.2f} "
                "seconds from playback start"
            )

            print(
                "Prompt-to-detection time: "
                f"{prompt_elapsed:.2f} seconds"
            )

            if playback.poll() is None:
                playback.terminate()

            break


finally:

    # If playback ended naturally,
    # make sure timeout thread can exit.
    if playback.poll() is not None:
        test_finished_event.set()

    try:
        playback.wait(
            timeout=2
        )

    except subprocess.TimeoutExpired:
        playback.kill()

    capture_stop_event.set()

    capture_thread.join(
        timeout=2
    )


# ============================================================
# Save microphone recording
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

capture_file = (
    ROOT
    / "test_audio"
    / f"barge_in_v4_webrtc_{timestamp}.wav"
)

if capture_chunks:

    complete_audio = np.concatenate(
        capture_chunks
    )

    with wave.open(
        str(capture_file),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(1)

        wav_file.setsampwidth(
            SAMPLE_WIDTH
        )

        wav_file.setframerate(
            SAMPLE_RATE
        )

        wav_file.writeframes(
            complete_audio
            .astype(np.int16)
            .tobytes()
        )


# ============================================================
# Final result
# ============================================================

if stop_detected:

    print("Playback interrupted.")

elif test_failed_event.is_set():

    print(
        "Playback stopped because "
        "the controlled test timed out."
    )

else:

    print(
        "Playback ended without "
        "detecting the command."
    )


print(
    "Dropped windows:",
    dropped_windows,
)

if capture_chunks:

    print(
        "Saved microphone capture:",
        capture_file,
    )
