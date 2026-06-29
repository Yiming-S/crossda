"""Shared fixtures: synthetic EEG sessions so tests need no real data."""

from __future__ import annotations

import numpy as np
import pytest


def make_synthetic_sessions(n_sessions=5, n_trials=40, n_ch=22, n_times=256, seed=0):
    """Deterministic synthetic motor-imagery sessions.

    Each session is a dict ``{"x": (n_trials, n_ch, n_times), "y", "id", "label"}``
    with a class-discriminative oscillation plus a per-session domain shift, so the
    feature extractors, distances and DA methods all have real structure to act on.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_times)
    sessions = []
    for s in range(n_sessions):
        mix = np.eye(n_ch) + 0.15 * rng.standard_normal((n_ch, n_ch))
        offset = 0.3 * rng.standard_normal((n_ch, 1))
        X, y = [], []
        for i in range(n_trials):
            cls = 1 if i % 2 == 0 else 2
            base = 0.7 * rng.standard_normal((n_ch, n_times))
            f = 10.0 if cls == 1 else 22.0
            chans = slice(0, 6) if cls == 1 else slice(6, 12)
            base[chans, :] += 0.32 * np.sin(2 * np.pi * f * t)[None, :]
            X.append(mix @ base + offset)
            y.append(cls)
        sessions.append({
            "x": np.asarray(X),
            "y": np.asarray(y),
            "id": s,
            "label": f"synthetic_session_{s}",
        })
    return sessions


@pytest.fixture
def synthetic_sessions():
    """A fixed list of five synthetic sessions (seed=0)."""
    return make_synthetic_sessions(seed=0)


@pytest.fixture
def patched_loader(monkeypatch, synthetic_sessions):
    """Patch the data loader so ``process_subject`` runs on synthetic sessions."""
    from crossda.core import workers
    monkeypatch.setattr(workers, "_load_subject_sessions",
                        lambda *a, **k: synthetic_sessions)
    return synthetic_sessions
