#!/usr/bin/env python3
"""List every callable a PyTorch checkpoint would import, without loading it.

    python3 pickle_audit.py base/ljspeech-2000.ckpt

A .ckpt/.pth is a zip holding data.pkl. pickletools only parses opcodes, so
nothing runs. Callables outside the tensor-rebuild allowlist are printed as
SUSPICIOUS: inspect them before loading the file with torch.load.
"""
import io
import pickletools
import sys
import zipfile

SAFE = {
    "collections.OrderedDict", "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter", "torch._utils._rebuild_tensor",
    "torch.FloatStorage", "torch.HalfStorage", "torch.LongStorage", "torch.IntStorage",
    "torch.BFloat16Storage", "torch.DoubleStorage", "torch.BoolStorage", "torch.ByteStorage",
    "torch.Size", "torch.float32", "torch.float16",
    "pathlib.PosixPath",  # Lightning stores paths in hyperparameters; builds a path, runs nothing
}


def callables(data: bytes) -> set:
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            found.add(arg.replace(" ", "."))
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            found.add(f"{strings[-2]}.{strings[-1]}")
    return found


def pickles(path):
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            if n.endswith(".pkl"):
                yield f"{path}!{n}", z.read(n)
    else:
        yield path, open(path, "rb").read()


bad_total = 0
for path in sys.argv[1:]:
    for name, data in pickles(path):
        g = callables(data)
        bad = sorted(x for x in g if x not in SAFE)
        bad_total += len(bad)
        print(f"{name}: {len(g)} callables; SUSPICIOUS: {bad or 'none'}")
sys.exit(1 if bad_total else 0)
