"""Fast checks of the bundled checkpoint and the transcript merge (CPU, a few seconds). Run: pytest tests/"""
import json
import os

import numpy as np
import pytest

from sed import Detector, rich_transcript, class_runs
from sed.model import FRAME, SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPTS = ["checkpoints"]


@pytest.mark.parametrize("ckpt", CKPTS)
@pytest.mark.parametrize("profile", ["sensitive", "balanced"])
def test_detect_schema(ckpt, profile):
    det = Detector.from_dir(os.path.join(ROOT, ckpt), profile=profile, device="cpu")
    rng = np.random.default_rng(0)
    audio = (0.05 * rng.standard_normal(int(11.3 * SR))).astype(np.float32)     # longer than one 8 s window
    times, probs = det.frame_probs(audio)
    assert probs.shape == (len(times), len(det.classes)) and np.all((probs >= 0) & (probs <= 1))
    for e in det.detect(audio):
        assert set(e) == {"label", "start", "end", "prob"} and e["label"] in det.classes
        assert 0 <= e["start"] < e["end"] <= len(audio) / SR + FRAME


@pytest.mark.parametrize("ckpt", CKPTS)
def test_safetensors_matches_best_pt(ckpt):
    d = os.path.join(ROOT, ckpt)
    thr = json.load(open(os.path.join(d, "thresholds.json")))
    a = Detector(os.path.join(d, "model.safetensors"), thr, device="cpu")
    b = Detector(os.path.join(d, "best.pt"), thr, device="cpu")
    x = (0.05 * np.random.default_rng(1).standard_normal(5 * SR)).astype(np.float32)
    assert a.classes == b.classes
    np.testing.assert_allclose(a.frame_probs(x)[1], b.frame_probs(x)[1], atol=1e-6)


def test_class_runs_hysteresis_and_gap():
    p = np.zeros(100); p[10:20] = 0.9; p[20:22] = 0.5; p[22:30] = 0.9; p[60:62] = 0.95
    assert class_runs(p, 0.8, low_ratio=0.5, gap=1.0, min_dur=0.15) == [(10, 30)]   # dip bridged; 2-frame blip dropped
    assert class_runs(p, 1.01) == []                                                    # threshold > 1 = off


def test_transcript_keeps_word_order():
    words = [("one", 0.0, 0.3), ("two", 0.5, 0.8), ("three", 0.4, 0.6), ("four", 1.5, 1.8), ("five", None, None)]
    out = rich_transcript(words, [dict(label="cough", start=0.45, end=0.7, prob=.9),
                                  dict(label="horn", start=1.0, end=2.0, prob=.9)])
    assert [t for t in out.split() if t[0] not in "[<"] == ["one", "two", "three", "four", "five"]
    assert "[cough]" in out and "<horn>" in out
