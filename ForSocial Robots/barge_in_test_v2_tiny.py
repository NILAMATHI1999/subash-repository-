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


ROOT = Path("/home/subash/thesis-social-robot")

PLAYBACK_FILE = (
    ROOT / "test_audio/barge_in_long_reply.wav"
)

CAPTURE_DEVICE = "hw:1,0"

PLAYBACK_TARGET = (
    "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__"
    "UAC1.0_-00.analog-stereo"
)

SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

CHUNK_SECONDS = 0.1
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SECONDS)
CHUNK_BYTES = CHUNK_FRAMES * CHANNELS * SAMPLE_WIDTH

WINDOW_SECONDS = 1.0
WINDOW_CHUNKS = int(WINDOW_SECONDS / CHUNK_SECONDS)

HOP_SECONDS = 0.25
HOP_FRAMES = int(SAMPLE_RATE * HOP_SECONDS)

QUEUE_SIZE = 2
SAY_STOP_AFTER_SECONDS = 3.0


def save_capture(chunks, output_file):
    with wave.open(str(output_file), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)

        audio = b"".join(
            chunk.tobytes()
            for chunk in chunks
        )

        output.writeframes(audio)


def main():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    capture_file = (
        ROOT
        / "test_audio"
        / f"barge_in_v2_{timestamp}.wav"
    )

    print("Loading Faster-Whisper...")

    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
    )

    print("Model ready.")

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

    recent_audio = deque(
        maxlen=WINDOW_CHUNKS
    )

    complete_audio = []

    window_queue = queue.Queue(
        maxsize=QUEUE_SIZE
    )

    audio_lock = threading.Lock()

    recording = True
    dropped_windows = 0
    captured_frames = 0
    next_window_frame = SAMPLE_RATE

    def read_microphone():
        nonlocal recording
        nonlocal dropped_windows
        nonlocal captured_frames
        nonlocal next_window_frame

        while recording:
            raw = recorder.stdout.read(
                CHUNK_BYTES
            )

            if len(raw) != CHUNK_BYTES:
                break

            samples = np.frombuffer(
                raw,
                dtype=np.int16,
            ).reshape(
                -1,
                CHANNELS,
            )

            channel_zero = samples[:, 0].copy()

            captured_frames += len(
                channel_zero
            )

            with audio_lock:
                recent_audio.append(
                    channel_zero
                )

                complete_audio.append(
                    channel_zero
                )

                if (
                    len(recent_audio)
                    < WINDOW_CHUNKS
                ):
                    continue

                if (
                    captured_frames
                    < next_window_frame
                ):
                    continue

                window = np.concatenate(
                    list(recent_audio)
                )

                next_window_frame += (
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

    reader = threading.Thread(
        target=read_microphone,
        daemon=True,
    )

    reader.start()

    time.sleep(0.5)

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

    prompt_time = [None]

    print("Playback started.")
    print("Wait for the SAY NOW message.")

    def prompt_user():
        time.sleep(
            SAY_STOP_AFTER_SECONDS
        )

        if playback.poll() is None:
            prompt_time[0] = (
                time.perf_counter()
            )

            print()
            print(
                ">>> SAY NOW: Stop robot"
            )

    prompt_thread = threading.Thread(
        target=prompt_user,
        daemon=True,
    )

    prompt_thread.start()

    stop_detected = False

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
                    "Heard:",
                    text,
                    "| inference:",
                    round(inference_time, 2),
                    "s",
                )

            if re.search(
                r"\bstop\b[\s,;:!\-]+\brobot\b",
                text,
                flags=re.IGNORECASE,
            ):
                if playback.poll() is not None:
                    break

                stop_detected = True

                detection_time = (
                    time.perf_counter()
                )

                print(
                    "STOP detected after",
                    round(
                        detection_time
                        - playback_started,
                        2,
                    ),
                    "seconds from playback start",
                )

                if prompt_time[0] is not None:
                    print(
                        "Prompt-to-detection time:",
                        round(
                            detection_time
                            - prompt_time[0],
                            2,
                        ),
                        "seconds",
                    )

                playback.terminate()

                try:
                    playback.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    playback.kill()
                    playback.wait()

                print(
                    "Playback interrupted."
                )

                break

        if not stop_detected:
            print(
                "STOP was not detected "
                "before playback finished."
            )

    finally:
        recording = False

        recorder.terminate()

        try:
            recorder.wait(timeout=2)
        except subprocess.TimeoutExpired:
            recorder.kill()
            recorder.wait()

        reader.join(timeout=1)

        with audio_lock:
            saved_audio = list(
                complete_audio
            )

        save_capture(
            saved_audio,
            capture_file,
        )

        print(
            "Dropped windows:",
            dropped_windows,
        )

        print(
            "Saved microphone capture:",
            capture_file,
        )


if __name__ == "__main__":
    main()
