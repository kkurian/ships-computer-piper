#!/usr/bin/env python3
"""Measure the voice on this machine: time to first audio, render speed and,
optionally, intelligibility (Whisper transcription of each line).

    python3 tools/bench.py [--runs 3] [--whisper]

Piper runs as a separate program (GPL-3.0; never imported). --whisper needs
`pip install transformers torch librosa` and downloads openai/whisper-base.en.
"""
import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import say  # noqa: E402  (stdlib only; finds the piper program and the voice)

LINES = [
    "Working.",
    "Unable to comply. The signing key is not available on this machine.",
    "Warning. Test suite failed. Two examples red in the landing queue.",
    "Specify parameters. Which branch should be merged into main?",
    "Accessing library computer data. Three files modified. All tests passing.",
    "Project keel. Build complete. Three of four features passing.",
    "Update to project keel. The landing queue processed four changes. Three passed "
    "their tests and were merged. One failed a regression check and was returned "
    "to the author.",
]


def synth(text: str):
    """Run piper once; return (seconds to first audio byte, total seconds, raw bytes)."""
    t0 = time.monotonic()
    p = subprocess.Popen([say.piper_bin(), "-m", str(say.MODEL), "--output-raw"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.stdin.write(text.encode())
    p.stdin.close()
    first, buf = None, bytearray()
    while chunk := p.stdout.read1(65536):
        if first is None:
            first = time.monotonic() - t0
        buf += chunk
    p.wait()
    return first, time.monotonic() - t0, bytes(buf)


def norm(s: str) -> list:
    return re.sub(r"[^a-z ]", "", s.lower()).split()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--whisper", action="store_true")
    a = ap.parse_args()
    rate = say.sample_rate()
    print(f"voice {say.MODEL.name}; load average {os.getloadavg()[0]:.1f}; {a.runs} runs per line\n")
    print(f"{'line':>4}  {'speech s':>8}  {'first audio s':>13}  {'total s':>7}  {'s per s of speech':>17}")
    tot_wall = tot_audio = 0.0
    firsts, wavs = [], []
    for i, text in enumerate(LINES, 1):
        results = [synth(text) for _ in range(a.runs)]
        first = sorted(r[0] for r in results)[len(results) // 2]   # median
        wall = sorted(r[1] for r in results)[len(results) // 2]
        audio = len(results[0][2]) / 2 / rate
        firsts.append(first)
        tot_wall += wall
        tot_audio += audio
        print(f"{i:>4}  {audio:>8.2f}  {first:>13.2f}  {wall:>7.2f}  {wall / audio:>17.2f}")
        f = Path(tempfile.gettempdir()) / f"bench_line{i}.wav"
        with wave.open(str(f), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes(results[0][2])
        wavs.append(f)
    print(f"\nall lines: {tot_wall:.1f} s to make {tot_audio:.1f} s of speech = "
          f"{tot_wall / tot_audio:.2f} s per second of speech; "
          f"median first audio {sorted(firsts)[len(firsts) // 2]:.2f} s (includes program start)")
    if a.whisper:
        import librosa
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
        scores = []
        print("\nintelligibility (Whisper base.en, word-sequence similarity to the input):")
        for text, f in zip(LINES, wavs):
            audio, _ = librosa.load(str(f), sr=16000, mono=True)
            got = asr({"raw": audio, "sampling_rate": 16000})["text"].strip()
            s = difflib.SequenceMatcher(None, norm(text), norm(got)).ratio()
            scores.append(s)
            print(f"  {s:.2f}  {got}")
        print(f"  mean {sum(scores) / len(scores):.2f}; exact {sum(x == 1.0 for x in scores)} of {len(scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
