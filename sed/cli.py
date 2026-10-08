"""Detect non-verbal sounds and background noise in audio, optionally merged into a Whisper transcript.

Examples
    python -m sed.cli call.mp3
    python -m sed.cli recordings/ --asr none --json events.jsonl
    python -m sed.cli call.wav --language hi --hide breath,lip_smack --times
    python -m sed.cli call.wav --thr cough=0.5,machine=1.01      # per-run threshold changes (above 1 = off)
    python -m sed.cli call.wav --profile balanced                 # fewer, more precise tags
    python -m sed.cli call.wav --model the-psyche/speech-event-detector           # weights from the Hugging Face Hub
"""
import argparse
import glob
import json
import os
import sys
import time

from .audio import SR, load_audio
from .detector import Detector, PROFILES
from .transcript import clean, rich_transcript

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm", ".mp4"}


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True)
                            if os.path.splitext(f)[1].lower() in AUDIO_EXT)
        elif os.path.exists(p):
            files.append(p)
        else:
            sys.exit(f"not found: {p}")
    return files


class ASR:
    """Word-timestamped transcription with openai-whisper, or Hugging Face transformers as a fallback."""

    def __init__(self, backend, size, language, device):
        self.language = language
        if backend == "auto":
            try:
                import whisper  # noqa: F401
                backend = "openai-whisper"
            except ImportError:
                backend = "transformers"
        self.backend = backend
        if backend == "openai-whisper":
            import whisper
            self.model = whisper.load_model(size, device=device)
        elif backend == "transformers":
            from transformers import pipeline
            self.model = pipeline("automatic-speech-recognition", model=f"openai/whisper-{size}",
                                  device=0 if str(device).startswith("cuda") else -1)
        else:
            raise ValueError(backend)

    def words(self, audio):
        if self.backend == "openai-whisper":
            r = self.model.transcribe(audio, word_timestamps=True, language=self.language,
                                      fp16=str(self.model.device).startswith("cuda"))
            return [(w["word"], w["start"], w["end"]) for s in r["segments"] for w in s.get("words", [])]
        kw = {"task": "transcribe"}
        if self.language: kw["language"] = self.language
        out = self.model(audio.copy(), return_timestamps="word", chunk_length_s=30, generate_kwargs=kw)
        return [(c["text"], c["timestamp"][0], c["timestamp"][1] if c["timestamp"][1] is not None else c["timestamp"][0] + 0.3)
                for c in out["chunks"] if c["timestamp"][0] is not None]


def default_model_dir():
    # The repo's checkpoints/ folder (GitHub layout), else the folder this package sits in (Hugging Face layout).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in (os.path.join(root, "checkpoints"), root):
        if any(os.path.exists(os.path.join(d, f)) for f in ("model.safetensors", "best.pt")):
            return d
    return os.path.join(root, "checkpoints")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="audio files or folders (any format ffmpeg reads)")
    ap.add_argument("--model", "--ckpt-dir", dest="model", default=None,
                    help="model folder (model.safetensors or best.pt + thresholds) or a Hugging Face repo id; "
                         "default: the bundled checkpoint")
    ap.add_argument("--profile", default="sensitive", choices=sorted(PROFILES),
                    help="sensitive: catch more sounds (more false tags); balanced: best F1 per class")
    ap.add_argument("--asr", default="auto", choices=["auto", "openai-whisper", "transformers", "none"],
                    help="transcription backend; 'none' = events only")
    ap.add_argument("--asr-model", default="small", help="Whisper size: tiny/base/small/medium/large-v3")
    ap.add_argument("--language", default=None, help="e.g. hi, en; default: auto-detect")
    ap.add_argument("--hide", default="", help="comma-separated tags to leave out, e.g. breath,lip_smack")
    ap.add_argument("--times", action="store_true", help="put start-end times inside the tags")
    ap.add_argument("--thr", default="", help="per-class threshold overrides, e.g. sigh=0.7,cough=0.5 (above 1 = off)")
    ap.add_argument("--min-prob", type=float, default=0.0, help="drop events whose mean probability is below this")
    ap.add_argument("--json", default=None, help="also write one JSON line per file here")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args(argv)

    files = collect(args.inputs)
    hide = {t.strip() for t in args.hide.split(",") if t.strip()}
    overrides = {k.strip(): float(v) for k, v in (kv.split("=") for kv in args.thr.split(",") if kv.strip())}
    det = Detector.from_pretrained(args.model or default_model_dir(), profile=args.profile,
                                   device=args.device, overrides=overrides)
    unknown = (set(overrides) | hide) - set(det.classes)
    if unknown:
        sys.exit(f"unknown classes: {sorted(unknown)}; known: {det.classes}")
    asr = None if args.asr == "none" else ASR(args.asr, args.asr_model, args.language, det.dev)
    out = open(args.json, "w") if args.json else None

    for path in files:
        t0 = time.time()
        audio = load_audio(path)
        events = [e for e in det.detect(audio) if e["label"] not in hide and e["prob"] >= args.min_prob]
        rec = dict(file=path, duration=round(len(audio) / SR, 2), events=events)
        print(f"\n=== {path}  ({rec['duration']}s)")
        if asr:
            words = asr.words(audio)
            events = clean(words, events)
            rec.update(events=events, words=[dict(word=w.strip(), start=s, end=e) for w, s, e in words],
                       transcript=rich_transcript(words, events, include_times=args.times))
            print(rec["transcript"])
        for e in events:
            print(f"  {e['start']:7.2f}-{e['end']:7.2f}s  {e['label']:<14} p={e['prob']:.2f}")
        if not events:
            print("  (no events)")
        print(f"  [{time.time() - t0:.1f}s]")
        if out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out:
        out.close()
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
