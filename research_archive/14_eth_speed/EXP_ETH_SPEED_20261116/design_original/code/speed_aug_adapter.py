"""Change ONLY the existing B0 augmentation bounds in an isolated process.

Uniform[0.5,2.0], probability .5; retain original reflection and torch RNG draw
order. Not an extra augmentation stacked on top of [0.9,1.1].
"""
from __future__ import annotations

import copy
from contextlib import contextmanager


@contextmanager
def wide_speed(factory):
    original = factory.AUGMENT
    updated = copy.copy(original)
    updated.augment_speed_min, updated.augment_speed_max = .5, 2.
    factory.AUGMENT = updated
    try:
        yield
    finally:
        factory.AUGMENT = original


def trainer_type(core, factory):
    class SpeedAugTrainer(core.Trainer):
        def step(self):
            with wide_speed(factory):
                return super().step()
    return SpeedAugTrainer
