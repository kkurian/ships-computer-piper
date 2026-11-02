"""Run piper.train with a simpler checkpoint policy.

Keeps last.ckpt (rewritten every epoch) and a snapshot every STAGE_EVERY
epochs (stage-<epoch>.ckpt, default 250). Piper's top-5 val_mel / val_mos
savers are dropped to keep disk use bounded.
"""
import os

import piper.train.__main__ as m
from lightning.pytorch.callbacks import ModelCheckpoint


def _callbacks():
    return [
        ModelCheckpoint(monitor=None, save_top_k=1, every_n_epochs=1,
                        filename="last", enable_version_counter=False),
        ModelCheckpoint(monitor=None, save_top_k=-1,
                        every_n_epochs=int(os.environ.get("STAGE_EVERY", "250")),
                        filename="stage-{epoch}", auto_insert_metric_name=False),
    ]


m._default_callbacks = _callbacks
m.main()
