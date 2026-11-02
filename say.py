#!/usr/bin/env python3
"""Speak text in the ship's-computer voice, streamed, or render it to a WAV file.

    say.py "Diagnostic complete."          # speak (stdin works too)
    say.py --out reply.wav "Working."      # render to a file instead

Piper (GPL-3.0) runs as a separate program; this file never imports it. Piper
writes raw 16-bit mono audio one sentence at a time, so playback starts as soon
as the first sentence is synthesized. The audio passes through light
compression and a limiter (VOICE_FX) on its way to the speakers.

Environment:
  SHIPS_COMPUTER_PIPER_BIN    the piper program (default: .piper/bin/piper
                              next to this file, else `piper` on PATH)
  SHIPS_COMPUTER_PIPER_MODEL  the voice (default: en_US-shipscomputer-high.onnx
                              next to this file)
  SHIPS_COMPUTER_VOICE_FX     ffmpeg audio filter chain; "" disables processing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path(os.environ.get("SHIPS_COMPUTER_PIPER_MODEL", HERE / "en_US-shipscomputer-high.onnx"))

# Moderate compression (threshold -20 dB, 4:1, 5 ms attack, 80 ms release),
# a fixed gain and a limiter: about -14 dB mean, peaks at -0.3 dBFS for this
# voice. Fixed gain rather than loudnorm, which cannot measure a live stream
# and misjudges one-word lines.
VOICE_FX = os.environ.get(
    "SHIPS_COMPUTER_VOICE_FX",
    "acompressor=threshold=-20dB:ratio=4:attack=5:release=80:makeup=6dB,"
    "volume=5dB,alimiter=limit=0.97:level=disabled",
)


def piper_bin() -> str:
    explicit = os.environ.get("SHIPS_COMPUTER_PIPER_BIN")
    if explicit:
        return explicit
    local = HERE / ".piper" / "bin" / "piper"
    if local.exists():
        return str(local)
    found = shutil.which("piper")
    if not found:
        sys.exit("piper not found: see README.md (Setup)")
    return found


def sample_rate() -> int:
    try:
        return int(json.loads(Path(f"{MODEL}.json").read_text())["audio"]["sample_rate"])
    except (OSError, ValueError, KeyError):
        return 22050


def _raw_in(rate: int) -> list[str]:
    return ["-f", "s16le", "-sample_rate", str(rate), "-ch_layout", "mono", "-i", "-"]


def _fx() -> list[str]:
    return ["-af", VOICE_FX] if VOICE_FX else []


def player_pipeline(rate: int) -> list[list[str]]:
    """Commands that read raw s16le mono on stdin, process it and play it."""
    if sys.platform == "darwin":
        # ffmpeg straight to Core Audio. Not ffplay: started before Piper's
        # first sentence arrives, ffplay clips the first syllable.
        return [["ffmpeg", "-loglevel", "quiet", *_raw_in(rate), *_fx(), "-f", "audiotoolbox", "-"]]
    play = ["paplay", "--raw", f"--rate={rate}", "--channels=1", "--format=s16le"]
    if not VOICE_FX:
        return [play]
    return [["ffmpeg", "-loglevel", "quiet", *_raw_in(rate), *_fx(), "-f", "s16le", "-"], play]


def _run(text: str, stages_cmds: list[list[str]]) -> bool:
    synth = subprocess.Popen([piper_bin(), "-m", str(MODEL), "--output-raw"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    synth.stdin.write(text.encode())
    synth.stdin.close()
    stages, upstream = [], synth.stdout
    for i, cmd in enumerate(stages_cmds):
        last = i == len(stages_cmds) - 1
        p = subprocess.Popen(cmd, stdin=upstream,
                             stdout=subprocess.DEVNULL if last else subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        upstream.close()  # the next stage owns the read end now
        upstream = p.stdout
        stages.append(p)
    err = synth.stderr.read()
    synth.wait()
    for p in stages:
        p.wait()
    ok = synth.returncode == 0 and all(p.returncode == 0 for p in stages)
    if not ok:
        sys.stderr.write(err.decode(errors="replace"))
    return ok


def speak(text: str) -> bool:
    """Stream `text` to the speakers. True when every stage exits cleanly."""
    return _run(text, player_pipeline(sample_rate()))


def render(text: str, out_wav: str) -> bool:
    """Write `text` as a processed WAV file."""
    rate = sample_rate()
    return _run(text, [["ffmpeg", "-loglevel", "quiet", "-y", *_raw_in(rate), *_fx(), out_wav]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="*", help="text to speak (default: stdin)")
    ap.add_argument("--out", help="render to this WAV file instead of speaking")
    a = ap.parse_args()
    text = " ".join(a.text).strip() or sys.stdin.read().strip()
    if not text:
        return 0
    if not MODEL.exists():
        sys.exit(f"voice not found: {MODEL}")
    return 0 if (render(text, a.out) if a.out else speak(text)) else 1


if __name__ == "__main__":
    sys.exit(main())
