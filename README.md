# ships-computer-piper

## TL;DR

A [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voice in the style of the Star Trek: The Next Generation ship's computer, plus a script that speaks with it, streamed. It runs on a CPU. On an M1 under load it makes speech about 3 times faster than real time (0.36 s of work per second of speech) and starts speaking about 1 s after it is called. Training took about 7.8 GPU-hours on one NVIDIA RTX PRO 4500 and cost about $6 on Runpod ($8.73 including a rejected second candidate).

## Background

Piper is a small, fast text-to-speech engine. A Piper voice is two files: an ONNX model and a JSON config. This voice was fine-tuned from Piper's public LJ Speech "high" voice on 882 transcribed lines of the TNG computer voice. The voice weights carry a separate notice: see [VOICE-NOTICE.md](VOICE-NOTICE.md).

## What's here

- `say.py`: speaks text, streamed, or renders a WAV file. Python 3.9+, standard library only.
- `requirements-piper.txt`: the Piper engine, pinned.
- `training/`: the scripts that made the voice (data preparation, GPU host setup, training, export).
- `tools/`: `bench.py` (speed and intelligibility), `inspect_onnx.py` and `pickle_audit.py` (safety checks for model files).
- The voice itself is not in git. It is attached to the [latest release](../../releases/latest).

## Install

macOS shown; on Linux use your package manager for `ffmpeg` and PulseAudio/PipeWire for `paplay`.

```sh
git clone https://github.com/kkurian/ships-computer-piper.git
cd ships-computer-piper
gh release download --pattern 'en_US-shipscomputer-high.onnx*' --pattern SHA256SUMS   # or download them from the latest release page
shasum -a 256 -c SHA256SUMS
python3 -m venv .piper && .piper/bin/pip install -r requirements-piper.txt
brew install ffmpeg
```

## Use

```sh
./say.py "Diagnostic complete."
echo "Working." | ./say.py
./say.py --out reply.wav "Unable to comply."
```

- `say.py` runs Piper as a separate program and pipes its raw audio through `ffmpeg` to the speakers. Piper writes one sentence at a time, so speech starts when the first sentence is ready.
- The audio passes through moderate compression, +5 dB and a limiter: about -14 dB mean, peaks at -0.3 dBFS. Set `SHIPS_COMPUTER_VOICE_FX` to change the `ffmpeg` filter chain, or to `""` to turn it off.
- On macOS the player is `ffmpeg -f audiotoolbox`, not `ffplay`. `ffplay` clipped the first syllable when it started before Piper's first sentence arrived.
- `SHIPS_COMPUTER_PIPER_BIN` and `SHIPS_COMPUTER_PIPER_MODEL` override where the engine and the voice are found.

## How it was made

Data
- Source: [STCOMP/majel_tts_medium_clean](https://huggingface.co/datasets/STCOMP/majel_tts_medium_clean), 1,005 transcribed clips, 2.00 h, 24 kHz. Its licence is not stated.
- `training/prepare_dataset.py` drops 123 clips (sources 509-633: interviews, documentary narration, an audiobook passage) and resamples to 22,050 Hz mono. Result: 882 clips, 1.53 h, mean 6.3 s.

Base voice
- Piper's LJ Speech "high" checkpoint, `ljspeech-2000.ckpt` from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints). LJ Speech is public domain.
- `tools/pickle_audit.py` run before loading: the checkpoint calls only tensor-rebuild functions, `OrderedDict` and `pathlib.PosixPath`.

Training
- Piper's trainer, [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) at commit `057093803b`, run by `training/train.sh` with `training/train_wrap.py`.
- Warm start from the base, batch 24, length bucketing on, 2,000 epochs (the learning rate anneals to 5% of its start by epoch 2,000), quality metric off. "High" architecture settings are passed explicitly; Piper's defaults are "medium".
- A snapshot was exported every 250 epochs. Epoch 2,000 was chosen by ear.
- A second candidate, Piper's Kristin "medium" voice, was trained for 200 epochs in parallel on an A100 and rejected by ear.

Hardware and time
- One NVIDIA RTX PRO 4500 Blackwell Server Edition (32 GB), 16 vCPUs, Runpod Secure Cloud, US-KS-2, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8, CUDA 12.8).
- About 14.4 s per epoch. Epochs 0-203: 53 min. Epochs 203-2,000: 6 h 55 min, resumed from `last.ckpt` on a second pod of the same type.

Cost, from Runpod's billing records, idle time excluded

| Pod | Use | Cost |
|---|---|---|
| RTX PRO 4500, $0.72/h | setup, epochs 0-203 | $0.95 |
| RTX PRO 4500, $0.72/h | epochs 203-2,000 | $5.24 |
| A100 SXM 80 GB, $1.79/h | rejected Kristin candidate, 200 epochs | $2.51 |
| L40S, $1.09/h | host with broken DNS, deleted after 2 min | $0.03 |
| 30 GB network volume | checkpoints during the first run | $0.01 |
| **Total** | | **$8.73** (this voice alone: $6.18) |

## Performance on an M1

Apple M1, 16 GB, macOS 26.5.2, load average 11.8 during the run (a busy machine). `tools/bench.py --runs 3 --whisper`, median of 3 runs. Times include starting the Piper program (about 0.8 s).

| Line | Speech | First audio | Total | Work per second of speech |
|---|---|---|---|---|
| "Working." | 0.82 s | 0.83 s | 0.85 s | 1.03 s |
| "Unable to comply. The signing key is not available on this machine." | 4.76 s | 1.01 s | 1.72 s | 0.36 s |
| "Warning. Test suite failed. Two examples red in the landing queue." | 5.18 s | 0.94 s | 1.91 s | 0.37 s |
| "Specify parameters. Which branch should be merged into main?" | 4.17 s | 1.02 s | 1.57 s | 0.38 s |
| "Accessing library computer data. Three files modified. All tests passing." | 5.65 s | 1.20 s | 2.16 s | 0.38 s |
| "Project keel. Build complete. Three of four features passing." | 4.70 s | 0.94 s | 1.72 s | 0.37 s |
| Three-sentence landing-queue update (29 words) | 10.50 s | 1.06 s | 3.11 s | 0.30 s |
| **All** | **35.8 s** | **1.01 s median** | **13.1 s** | **0.36 s** |

Intelligibility: Whisper base.en transcribed each line; the score is word-sequence similarity to the input (1.00 = exact).

| Line | Whisper heard | Score |
|---|---|---|
| "Working." | "Work in." | 0.00 |
| "Unable to comply…" | "Unable to comply, the sign-in key is not available on this machine." | 0.92 |
| "Warning. Test suite failed…" | "Warning, tests wait failed. Two examples read in the landing queue." | 0.73 |
| "Specify parameters…" | "Specify parameters, which branch should be merged into main." | 1.00 |
| "Accessing library computer data…" | "accessing library computer data, three files modified, all tests passing." | 1.00 |
| "Project keel…" | "Project Kale. Build complete. Three of four features passing." | 0.89 |
| Landing-queue update | exact | 1.00 |

Mean 0.79; 3 of 7 exact. Single words are weakest ("Working." → "Work in"). "red/read" counts as an error even though the audio is correct.

Not measured: GPU inference, Linux, older Macs, and listener ratings.

## Licence

- Code and documentation: MIT, see [LICENSE](LICENSE).
- Voice weights (`en_US-shipscomputer-high.onnx` and its `.json`): not MIT. See [VOICE-NOTICE.md](VOICE-NOTICE.md).
- Piper (`piper-tts`) is GPL-3.0. `say.py` and `tools/bench.py` run it as a separate program and never import it.
- No sound effects or third-party audio are included.
