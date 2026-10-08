"""Waveform -> timestamped events, using a trained checkpoint and per-class thresholds."""
import json
import os

import numpy as np
import torch
from scipy.ndimage import median_filter, find_objects, label as label_runs

from .audio import load_audio
from .model import SR, FRAME, DUR, LogMel, SED
from .taxonomy import INLINE

PROFILES = {"sensitive": "thresholds_sensitive.json", "balanced": "thresholds.json"}


def class_runs(p, hi, low_ratio=0.6, gap=1.0, min_dur=0.15):
    # Event frames [start, stop) for one class. Hysteresis: a run starts above `hi` and continues while
    # above low_ratio*hi, so short dips don't split one horn into five. Runs closer than `gap` seconds
    # are joined; runs shorter than `min_dur` are dropped. hi > 1 means the class is switched off.
    if hi > 1 or p.max() <= hi: return []
    lab, _ = label_runs(p > low_ratio * hi)
    merged = []
    for (sl,) in find_objects(lab):
        if not (p[sl] > hi).any(): continue
        if merged and (sl.start - merged[-1][1]) * FRAME <= gap: merged[-1][1] = sl.stop
        else: merged.append([sl.start, sl.stop])
    return [(s, e) for s, e in merged if (e - s) * FRAME >= min_dur]


def load_weights(path, device="cpu"):
    # -> (state_dict, classes). Accepts the training checkpoint (best.pt: {"model", "classes"}) or
    # model.safetensors with its config.json next to it (the format published on the Hugging Face Hub).
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        return load_file(path, device=str(device)), cfg["classes"]
    ck = torch.load(path, map_location=device, weights_only=True)
    return ck["model"], ck["classes"]


class Detector:
    def __init__(self, weights, thresholds, device=None, low_ratio=0.6, batch=16):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state, self.classes = load_weights(weights, device)
        self.dev, self.low_ratio, self.batch = device, low_ratio, batch
        self.model = SED(len(self.classes)).to(device).eval(); self.model.load_state_dict(state)
        self.feats = LogMel().to(device)
        self.thr = np.array([thresholds.get(c, 0.5) for c in self.classes])

    @classmethod
    def from_dir(cls, ckpt_dir, profile="sensitive", device=None, overrides=None):
        # Loads the weights (model.safetensors, else best.pt) and one threshold profile from a folder:
        #   "sensitive": thresholds_sensitive.json, tuned for F2 (catches more, more false tags)
        #   "balanced":  thresholds.json, tuned for F1
        # threshold_overrides.json (manual per-class changes, kept when re-tuning) and then `overrides`
        # ({class: threshold}; above 1 = off) are applied on top. detect_config.json sets the hysteresis ratio.
        thr = json.load(open(os.path.join(ckpt_dir, PROFILES[profile])))
        ov = os.path.join(ckpt_dir, "threshold_overrides.json")
        if os.path.exists(ov):
            thr.update(json.load(open(ov)).get(profile, {}))
        thr.update(overrides or {})
        cfg = os.path.join(ckpt_dir, "detect_config.json")
        low_ratio = thr.pop("_low_ratio", json.load(open(cfg)).get("low_ratio", 0.6) if os.path.exists(cfg) else 0.6)
        st = os.path.join(ckpt_dir, "model.safetensors")
        weights = st if os.path.exists(st) else os.path.join(ckpt_dir, "best.pt")
        return cls(weights, thr, device=device, low_ratio=low_ratio)

    @classmethod
    def from_pretrained(cls, name_or_path, profile="sensitive", device=None, overrides=None, revision=None):
        # A local folder, or a Hugging Face model repo id (downloads only the weights, config and thresholds).
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download
            name_or_path = snapshot_download(name_or_path, revision=revision,
                                             allow_patterns=["*.safetensors", "*.json"])
        return cls.from_dir(name_or_path, profile=profile, device=device, overrides=overrides)

    @torch.no_grad()
    def frame_probs(self, x, win_s=DUR, hop_s=2.0):
        # x: 1-D float array/tensor at 16 kHz -> (times, probs[n_frames, n_classes]) on a 40 ms grid.
        # 8 s windows every 2 s; overlapping predictions are averaged.
        x = torch.as_tensor(x, dtype=torch.float32)
        win, hop = int(win_s * SR), int(hop_s * SR)
        n_tok = int(len(x) / SR / FRAME) + 1
        acc, cnt = np.zeros((n_tok, len(self.classes))), np.zeros(n_tok)
        starts = list(range(0, max(1, len(x) - win + hop), hop))
        for i in range(0, len(starts), self.batch):
            chunk = starts[i: i + self.batch]
            segs = torch.stack([torch.nn.functional.pad(x[t0: t0 + win], (0, win - len(x[t0: t0 + win])))
                                for t0 in chunk]).to(self.dev)
            P = torch.sigmoid(self.model(self.feats(segs))).float().cpu().numpy()
            for t0, p in zip(chunk, P):
                g = t0 // int(FRAME * SR) + np.arange(len(p)); ok = g < n_tok
                np.add.at(acc, g[ok], p[ok]); np.add.at(cnt, g[ok], 1)
        acc /= np.maximum(cnt, 1)[:, None]
        return np.arange(n_tok) * FRAME, acc

    def detect(self, audio, min_dur=0.15, gap_inline=0.30, gap_ambient=1.0, low_ratio=None):
        # audio: 1-D float array at 16 kHz, or a path to any file ffmpeg can read.
        # Returns [{'label', 'start', 'end', 'prob'}, ...] sorted by start time (seconds).
        low_ratio = self.low_ratio if low_ratio is None else low_ratio
        if isinstance(audio, (str, os.PathLike)):
            audio = load_audio(audio)
        _, probs = self.frame_probs(audio)
        probs = median_filter(probs, size=(5, 1))                     # de-jitter
        events = []
        for ci, name in enumerate(self.classes):
            gap = gap_inline if name in INLINE else gap_ambient
            for s, e in class_runs(probs[:, ci], self.thr[ci], low_ratio, gap, min_dur):
                events.append(dict(label=name, start=round(float(s * FRAME), 2), end=round(float(e * FRAME), 2),
                                   prob=round(float(probs[s:e, ci].mean()), 2)))
        return sorted(events, key=lambda e: e["start"])
