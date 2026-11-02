#!/usr/bin/env python3
"""Turn the STCOMP/majel_tts_medium_clean parquet into a Piper training set.

    python3 prepare_dataset.py train-00000-of-00001.parquet out/

Writes out/wavs/majel_NNNN.wav (22050 Hz mono 16-bit) and out/metadata.csv
(`file|text`). Rows 509-633 by `source` are dropped: interviews, documentary
narration and an audiobook passage, not the ship's-computer voice.

Needs: pip install pyarrow soundfile soxr numpy
"""
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

DROP = range(509, 634)  # sources 509..633 inclusive
RATE = 22050


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    rows, durs, dropped = [], [], 0
    for r in pq.read_table(src).to_pylist():
        s = int(r["source"])
        if s in DROP:
            dropped += 1
            continue
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = soxr.resample(audio, sr, RATE)
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            audio = audio / peak
        name = f"majel_{s:04d}.wav"
        sf.write(out / "wavs" / name, audio, RATE, subtype="PCM_16")
        text = " ".join(r["text"].split()).replace("|", " ")
        rows.append(f"{name}|{text}")
        durs.append(len(audio) / RATE)
    (out / "metadata.csv").write_text("\n".join(rows) + "\n")
    print(f"kept {len(rows)} clips, dropped {dropped}; {sum(durs) / 3600:.2f} h; "
          f"mean {sum(durs) / len(durs):.2f} s; max {max(durs):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
