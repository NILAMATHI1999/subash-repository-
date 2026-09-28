#!/usr/bin/env python3

import re
import subprocess
import threading
import time
import wave
from collections import deque

import numpy as np
from faster_whisper import WhisperModel


PLAYBACK_FILE = (
    "/home/subash/thesis-social-robot/"
    "test_audio/barge_in_long_reply.wav"
)

CAPTURE_FILE = (
    "/home/subash/thesis-social-robot/"
    "test_audio/barge_in_capture.wav"
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

WINDOW_CHUNKS = 10
CHECK_INTERVAL = 0.25


def save_capture(chunks):
    with wave.open(CAPTURE_FILE, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)

        audio = b"".join(
            chunk.tobytes()
            for chunk in chunks
        )

        output.writeframes(audio)


def main():
    print("Loading Faster-Whisper...")

    model = WhisperModel(
        "base",
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
    audio_lock = threading.Lock()
    recording = True

    def read_microphone():
        nonlocal recording

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

            channel_zero = (
                samples[:, 0].copy()
            )

            with audio_lock:
                recent_audio.append(
                    channel_zero
                )

                complete_audio.append(
                    channel_zero
                )

    reader = threading.Thread(
        target=read_microphone,
        daemon=True,
    )

    reader.start()
    time.sleep(0.5)

    print("Playback started.")
    print(
        "Say 'STOP ROBOT' while "
        "the robot is speaking."
    )

    playback = subprocess.Popen(
        [
            "pw-play",
            "--target",
            PLAYBACK_TARGET,
            PLAYBACK_FILE,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    stop_detected = False
    playback_started = time.perf_counter()

    try:
        while playback.poll() is None:
            time.sleep(
                CHECK_INTERVAL
            )

            with audio_lock:
                if (
                    len(recent_audio)
                    < WINDOW_CHUNKS
                ):
                    continue

                window = np.concatenate(
                    list(recent_audio)
                )

            audio = (
                window.astype(np.float32)
                / 32768.0
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

            if text:
                print("Heard:", text)

            if re.search(
                r"\bstop\b",
                text,
                flags=re.IGNORECASE,
            ):
                stop_detected = True

                detection_time = (
                    time.perf_counter()
                    - playback_started
                )

                print(
                    "STOP detected after",
                    round(detection_time, 2),
                    "seconds",
                )

                playback.terminate()
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
        recorder.wait()
        reader.join(timeout=1)

        with audio_lock:
            saved_audio = list(
                complete_audio
            )

        save_capture(saved_audio)

        print(
            "Saved microphone capture:",
            CAPTURE_FILE,
        )


if __name__ == "__main__":
    main()
