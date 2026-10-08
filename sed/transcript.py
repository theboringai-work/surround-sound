"""Merge detected events into ASR word timestamps: [tag] inline for speaker sounds, <tag> ... </tag> for background."""
import bisect
import collections

from .taxonomy import INLINE

FILLER_WORDS = {"uh", "um", "umm", "hmm", "ah", "er", "mhm"}

LAUGH_WORDS  = {"ha", "haha", "hehe", "heh", "[laughs]", "lol"}


def clean(words, events):
    # Drop an event the ASR already transcribed (e.g. a laugh written out as "haha").
    kept = []
    for ev in events:
        overlap = [w for w, s, e in words if s < ev["end"] and e > ev["start"]]
        if overlap and all(w.lower().strip(" .,!?") in FILLER_WORDS | LAUGH_WORDS for w in overlap):
            continue
        kept.append(ev)
    return kept

def rich_transcript(words, events, include_times=False, join_inline=0.5, join_ambient=1.5):
    # words: [(word, start_s, end_s), ...]   events: output of Detector.detect
    words = [(w.strip(), s, e) for w, s, e in words if w.strip()]
    # 1) join same-label pieces that are close together (one horn, not five)
    merged = []
    for ev in sorted(events, key=lambda e: (e["label"], e["start"])):
        gap = join_inline if ev["label"] in INLINE else join_ambient
        if merged and merged[-1]["label"] == ev["label"] and ev["start"] - merged[-1]["end"] <= gap:
            merged[-1]["end"] = max(merged[-1]["end"], ev["end"])
        else:
            merged.append(dict(ev))
    # Words stay in ASR order; tags are slotted between them by time. Word starts are made
    # non-decreasing first, so out-of-order ASR timestamps can never reshuffle the text.
    starts, t_prev = [], 0.0
    for _, s, _ in words:
        t_prev = max(t_prev, s if s is not None else t_prev); starts.append(t_prev)
    last_word_end = max((e if e is not None else s or 0) for _, s, e in words) if words else 0
    items = []                                             # (time, priority, text)
    for ev in merged:
        core = f"{ev['label']} {ev['start']:.2f}-{ev['end']:.2f}" if include_times else ev["label"]
        if ev["label"] in INLINE:                               # [tag] where the sound starts
            items.append((ev["start"], 0, f"[{core}]"))
        else:
            spans_words = any((s or 0) < ev["end"] and (e if e is not None else s or 0) > ev["start"] for _, s, e in words)
            items.append((ev["start"], 0, f"<{core}>"))
            if spans_words and ev["end"] + 0.3 < last_word_end:   # wrap words; a lone <tag> otherwise
                items.append((ev["end"], 2, f"</{ev['label']}>"))
    slots = collections.defaultdict(list)
    for t, prio, text in items:                            # prio 0: before words at t, 2: after them
        slot = bisect.bisect_left(starts, t) if prio == 0 else bisect.bisect_right(starts, t)
        slots[slot].append((round(t, 3), prio, text))
    seq = []
    for i in range(len(words) + 1):
        seq += [text for _, _, text in sorted(slots.get(i, []))]
        if i < len(words): seq.append(words[i][0])
    # 2) drop a tag that repeats the previous token ("[breath] [breath]" -> "[breath]")
    out = []
    for t in seq:
        if out and t == out[-1] and t[:1] in "[<": continue
        out.append(t)
    return " ".join(out).strip()
