# Data

Experiment 1 draws on seven public datasets. All numbers below are recomputed from the training run's cached index and audio banks; `paper/data/stats.json` has the raw values.

## Sources

| dataset | size (full) | labels | how it is used | split (held-out part) | license |
|---|---|---|---|---|---|
| [NonVerbalSpeech-38K](https://huggingface.co/datasets/nonverbalspeech/nonverbalspeech38k) ([paper](https://arxiv.org/abs/2508.05385)) | 38,718 clips, 130.6 h; 36,925 Mandarin, 1,793 English; radio dramas, comedy, cartoons, audiobooks… | one non-verbal event per clip, `non_verbal_region` = [start, end], 10 types | event crops; speech beds (audio away from the event); whole clips as real windows | 10% by a stable hash of (file, row group, row): 3,828 clips | CC BY-NC 4.0 |
| [Vaani Noise Event Timestamps](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) (gated; [blog post](https://huggingface.co/blog/ARTPARK-IISc/a-real-world-dataset-for-noise-robust-speech-ai)) | 90,637 clips, 154.6 h; 58 Indian languages plus English (Hindi 47k, Telugu 10k, Bengali 9k, …); 162 districts | 319 free-text tags with start/end; 11,111 verified, 61,642 unverified, 17,884 without timestamps | event crops; whole clips as real windows (only if every tag maps to a class) | 14 of 162 districts held out; only their verified clips are used for validation | CC BY 4.0 |
| [NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS) | 6,256 clips, 17.6 h, English (VoxCeleb 4,452, Expresso 1,804) | 8,620 non-verbal sounds as emoji inside the transcript, no timestamps | forced alignment gives timestamps (below); real windows, crops, English speech beds | official dev + test (405 clips) | Hub tag Apache-2.0; card: annotations CC BY-NC-SA 4.0, audio under the source corpora's terms |
| [AudioSet-Strong mirror](https://huggingface.co/datasets/enyoukai/AudioSet-Strong) | 93,609 10 s clips (79,564 train, 14,045 test), 260 h, 121 GB | 455 event names with start/end (temporally strong labels) | streamed; a greedy pass picks up to 500 training and 60 evaluation row groups (72 clips each) that best cover our classes; they give real windows and crops | official evaluation (test) split | not stated on the mirror |
| [OpenSLR 99](https://www.openslr.org/99/) (Deeply Nonverbal Vocalization, public set) | 727 clips, 410 speakers, 16 classes, phone recordings | one isolated vocalization per clip | trimmed to the vocalization (energy-based), event crops | 15% of speakers | CC BY-NC-ND 4.0 |
| [DNC](https://doi.org/10.14279/depositonce-11645) (TU Berlin) | 4,377 home background recordings | type of noise (file name) | TV/radio → `tv_speaker`, cars → `vehicle`, music → `music`, appliances → `machine`; quiet rooms and people → noise pool | 10% by recording session | MIT |
| [DEMAND](https://doi.org/10.5281/zenodo.1227121) | 18 environments × 16 channels × ~5 min | environment | channels 1 and 9; traffic/bus/car/metro → `vehicle`, washing room → `machine`; other environments → noise pool | last 20% of each recording | CC BY 4.0 (the record's text also states CC BY-SA 3.0) |

## Label mapping

Every source maps to one canonical tag set (`sed/taxonomy.py`, and `AS_RULES`/`DNC_MAP`/`DEMAND_MAP` in `training/build_notebooks.py`):
- **Vaani:** 319 free-text tags pass through ordered regex rules. Of the 106,892 tag occurrences, 99,320 map to a class, 7,502 are speech tags and are ignored (e.g. `<child talking>`), and 70 are odd tags (e.g. `<clicking>`, `<burping>`). A clip with an odd tag isn't used as a real window.
- **AudioSet:** 220 of the 455 event names map to a class. The rest (speech, generic impacts, …) are not our classes.
- **NonverbalTTS:** emoji → class. Grunt merges into `groan`.
- **OpenSLR 99:** folder name → class. Panting → `breath`, lip-popping → `lip_smack`, moaning → `groan`; teeth sounds are dropped.

55 events are listed. A class is trained if it has at least 80 events across all sources, which drops only `click` (47). That leaves **54 classes: 20 speaker sounds and 34 background sounds**.

### Events per class and dataset (whole index, before any caps)
| class | type | nonverbal38k | vaani | nonverbaltts | audioset | openslr99 | dnc | demand | total |
|---|---|---|---|---|---|---|---|---|---|
| `breath` | speaker | 2162 | 35663 | 6121 | 27232 | 44 | 0 | 0 | 71222 |
| `music` | background | 0 | 6872 | 0 | 51454 | 0 | 229 | 0 | 58555 |
| `bird` | background | 0 | 17694 | 0 | 37443 | 0 | 0 | 0 | 55137 |
| `vehicle` | background | 0 | 4854 | 0 | 25450 | 0 | 692 | 4 | 31000 |
| `wind` | background | 0 | 0 | 0 | 25079 | 0 | 0 | 0 | 25079 |
| `laugh` | speaker | 6912 | 99 | 1278 | 14489 | 25 | 0 | 0 | 22803 |
| `horn` | background | 0 | 15739 | 0 | 3794 | 0 | 0 | 0 | 19533 |
| `animal` | background | 0 | 1577 | 0 | 15820 | 0 | 0 | 0 | 17397 |
| `child_voice` | background | 0 | 3165 | 0 | 11447 | 0 | 0 | 0 | 14612 |
| `footsteps` | background | 0 | 0 | 0 | 12490 | 0 | 0 | 0 | 12490 |
| `dog_bark` | background | 0 | 2005 | 0 | 10429 | 0 | 0 | 0 | 12434 |
| `sigh` | speaker | 9783 | 0 | 147 | 206 | 51 | 0 | 0 | 10187 |
| `applause` | background | 0 | 0 | 0 | 10157 | 0 | 0 | 0 | 10157 |
| `water` | background | 0 | 0 | 0 | 10053 | 0 | 0 | 0 | 10053 |
| `beep` | background | 0 | 1039 | 0 | 8435 | 0 | 0 | 0 | 9474 |
| `machine` | background | 0 | 156 | 0 | 7379 | 0 | 853 | 1 | 8389 |
| `cough` | speaker | 5940 | 202 | 272 | 1818 | 50 | 0 | 0 | 8282 |
| `sniff` | speaker | 7321 | 32 | 407 | 473 | 0 | 0 | 0 | 8233 |
| `insect` | background | 0 | 3324 | 0 | 4561 | 0 | 0 | 0 | 7885 |
| `shout` | speaker | 0 | 0 | 0 | 6244 | 0 | 0 | 0 | 6244 |
| `siren` | background | 0 | 129 | 0 | 6023 | 0 | 0 | 0 | 6152 |
| `bell` | background | 0 | 668 | 0 | 4934 | 0 | 0 | 0 | 5602 |
| `throat_clear` | speaker | 4909 | 96 | 214 | 278 | 54 | 0 | 0 | 5551 |
| `construction` | background | 0 | 0 | 0 | 5400 | 0 | 0 | 0 | 5400 |
| `gunshot` | background | 0 | 0 | 0 | 5074 | 0 | 0 | 0 | 5074 |
| `whistle` | background | 0 | 106 | 0 | 4061 | 0 | 0 | 0 | 4167 |
| `dishes` | background | 0 | 0 | 0 | 3654 | 0 | 0 | 0 | 3654 |
| `baby_cry` | background | 0 | 1620 | 0 | 1576 | 0 | 0 | 0 | 3196 |
| `crowd` | background | 0 | 0 | 0 | 3130 | 0 | 0 | 0 | 3130 |
| `cheer` | background | 0 | 0 | 0 | 3118 | 0 | 0 | 0 | 3118 |
| `tv_speaker` | background | 0 | 156 | 0 | 1804 | 0 | 1100 | 0 | 3060 |
| `phone_ring` | background | 0 | 1548 | 0 | 1409 | 0 | 0 | 0 | 2957 |
| `explosion` | background | 0 | 0 | 0 | 2505 | 0 | 0 | 0 | 2505 |
| `cry` | speaker | 753 | 0 | 0 | 1234 | 24 | 0 | 0 | 2011 |
| `gasp` | speaker | 578 | 1 | 0 | 1166 | 0 | 0 | 0 | 1745 |
| `chew` | speaker | 0 | 0 | 0 | 1730 | 0 | 0 | 0 | 1730 |
| `rain` | background | 0 | 0 | 0 | 1711 | 0 | 0 | 0 | 1711 |
| `lip_smack` | speaker | 0 | 1437 | 0 | 0 | 99 | 0 | 0 | 1536 |
| `whisper` | speaker | 0 | 0 | 0 | 1470 | 0 | 0 | 0 | 1470 |
| `scream` | speaker | 0 | 0 | 0 | 1330 | 51 | 0 | 0 | 1381 |
| `snore` | speaker | 91 | 12 | 15 | 1101 | 0 | 0 | 0 | 1219 |
| `glass_break` | background | 0 | 0 | 0 | 1159 | 0 | 0 | 0 | 1159 |
| `groan` | speaker | 0 | 0 | 148 | 901 | 45 | 0 | 0 | 1094 |
| `fan` | background | 0 | 599 | 0 | 452 | 0 | 0 | 0 | 1051 |
| `buzzer` | background | 0 | 76 | 0 | 839 | 0 | 0 | 0 | 915 |
| `door` | background | 0 | 0 | 0 | 895 | 0 | 0 | 0 | 895 |
| `burp` | speaker | 0 | 0 | 0 | 765 | 0 | 0 | 0 | 765 |
| `thunder` | background | 0 | 0 | 0 | 728 | 0 | 0 | 0 | 728 |
| `hiccup` | speaker | 0 | 0 | 0 | 608 | 0 | 0 | 0 | 608 |
| `sneeze` | speaker | 0 | 87 | 18 | 440 | 52 | 0 | 0 | 597 |
| `knock` | background | 0 | 0 | 0 | 431 | 0 | 0 | 0 | 431 |
| `phone_vibrate` | background | 0 | 231 | 0 | 176 | 0 | 0 | 0 | 407 |
| `yawn` | speaker | 269 | 36 | 0 | 19 | 48 | 0 | 0 | 372 |
| `nose_blow` | speaker | 0 | 97 | 0 | 0 | 48 | 0 | 0 | 145 |

## NonverbalTTS: from emoji to timestamps

NonverbalTTS marks each sound with an emoji between two words of the transcript, but gives no timing:
1. Each clip is force-aligned with torchaudio's MMS_FA model. A CTC Viterbi aligner written in plain PyTorch gives word spans identical to torchaudio's aligner, with `*` wildcards at both ends for untranscribed speech.
2. Each sound is placed in the pause between the word before its emoji and the word after. The event is the most energetic stretch above the clip's noise floor + 4 dB, with a per-class maximum length.
3. Clips whose alignment score is below 0.35 are skipped. So are sounds whose pause is shorter than about 0.2 s, for example laughing while talking.

Result: **6,360 of 8,620 sounds (74%) were located**. 4,549 clips became real windows: 3,297 in which every sound was located, and 1,252 without any sound. 1,243 of the latter (those of at least 2 s) also became English speech beds. Clips longer than 30 s are not aligned.

## Audio banks (what training actually used)

Banks are int16, 16 kHz, memory-mapped. Per-class caps (about 3,000 training crops per class per source, checked per clip so AudioSet-Strong slightly exceeds it for a few classes; NonverbalTTS uncapped; real-window caps per source) keep common classes from dominating.

| store | items | hours | GB | split |
|---|---|---|---|---|
| event crops: NV38k + Vaani | 52,654 | 25.2 | 2.91 | nv/train: 18,475, nv/val: 2,253, vaani/train: 31,021, vaani/val: 905 |
| event crops: AudioSet | 75,343 | 25.9 | 2.98 | audioset/train: 67,102, audioset/val: 8,241 |
| event crops: NonverbalTTS (aligned) | 6,360 | 1.0 | 0.11 | nvtts/train: 5,856, nvtts/val: 504 |
| event crops: OpenSLR 99 | 591 | 0.4 | 0.04 | slr99/train: 503, slr99/val: 88 |
| ambient crops: DNC + DEMAND | 3,234 | 12.3 | 1.42 | demand/train: 300, demand/val: 70, dnc/train: 2,613, dnc/val: 251 |
| speech beds: NV38k | 8,862 | 18.0 | 2.07 | nv/train: 7,872, nv/val: 990 |
| speech beds: NonverbalTTS | 1,243 | 2.6 | 0.30 | nvtts/train: 1,243 |
| real windows: NV38k + Vaani | 37,434 | 80.6 | 9.29 | nv/train: 15,000, nv/val: 1,875, vaani/train: 19,999, vaani/val: 560 |
| real windows: AudioSet | 33,750 | 92.5 | 10.66 | audioset/train: 30,000, audioset/val: 3,750 |
| real windows: NonverbalTTS | 4,549 | 11.8 | 1.35 | nvtts/train: 4,273, nvtts/val: 276 |
| background-noise pool: DNC + DEMAND | 2,279 | 6.3 | 0.73 | demand/-: 780, dnc/-: 1,499 |

The speech beds and the noise pool were screened with a preliminary 31-class checkpoint of the same architecture: stretches where it detected a sound (±0.3 s) were cut, and the longest clean stretch of at least 2 s was kept. That left **9,645 of 10,105 beds (18.4 h)** and **2,102 of 2,279 noise chunks (5.5 h)**.

### Event crops per class and source
| class | nonverbal38k | vaani | audioset | nonverbaltts | openslr99 | dnc | demand | total |
|---|---|---|---|---|---|---|---|---|
| `breath` | 2162 | 3375 | 3376 | 4546 | 44 | 0 | 0 | 13503 |
| `laugh` | 3375 | 63 | 3378 | 879 | 25 | 0 | 0 | 7720 |
| `vehicle` | 0 | 3028 | 3377 | 0 | 0 | 692 | 296 | 7393 |
| `music` | 0 | 3005 | 3379 | 0 | 0 | 229 | 0 | 6613 |
| `bird` | 0 | 3169 | 3384 | 0 | 0 | 0 | 0 | 6553 |
| `child_voice` | 0 | 2874 | 3377 | 0 | 0 | 0 | 0 | 6251 |
| `dog_bark` | 0 | 1820 | 3382 | 0 | 0 | 0 | 0 | 5202 |
| `horn` | 0 | 3185 | 1661 | 0 | 0 | 0 | 0 | 4846 |
| `animal` | 0 | 1459 | 3383 | 0 | 0 | 0 | 0 | 4842 |
| `cough` | 3375 | 193 | 885 | 224 | 50 | 0 | 0 | 4727 |
| `insect` | 0 | 2564 | 1621 | 0 | 0 | 0 | 0 | 4185 |
| `sniff` | 3375 | 29 | 287 | 327 | 0 | 0 | 0 | 4018 |
| `beep` | 0 | 937 | 2940 | 0 | 0 | 0 | 0 | 3877 |
| `throat_clear` | 3375 | 89 | 167 | 187 | 54 | 0 | 0 | 3872 |
| `sigh` | 3375 | 0 | 134 | 107 | 51 | 0 | 0 | 3667 |
| `footsteps` | 0 | 0 | 3375 | 0 | 0 | 0 | 0 | 3375 |
| `wind` | 0 | 0 | 3371 | 0 | 0 | 0 | 0 | 3371 |
| `machine` | 0 | 93 | 2177 | 0 | 0 | 846 | 74 | 3190 |
| `applause` | 0 | 0 | 2613 | 0 | 0 | 0 | 0 | 2613 |
| `water` | 0 | 0 | 2510 | 0 | 0 | 0 | 0 | 2510 |
| `shout` | 0 | 0 | 2481 | 0 | 0 | 0 | 0 | 2481 |
| `bell` | 0 | 586 | 1836 | 0 | 0 | 0 | 0 | 2422 |
| `gunshot` | 0 | 0 | 2390 | 0 | 0 | 0 | 0 | 2390 |
| `siren` | 0 | 113 | 2259 | 0 | 0 | 0 | 0 | 2372 |
| `baby_cry` | 0 | 1436 | 890 | 0 | 0 | 0 | 0 | 2326 |
| `phone_ring` | 0 | 1462 | 702 | 0 | 0 | 0 | 0 | 2164 |
| `whistle` | 0 | 94 | 1736 | 0 | 0 | 0 | 0 | 1830 |
| `construction` | 0 | 0 | 1828 | 0 | 0 | 0 | 0 | 1828 |
| `tv_speaker` | 0 | 100 | 306 | 0 | 0 | 1097 | 0 | 1503 |
| `cry` | 753 | 0 | 649 | 0 | 24 | 0 | 0 | 1426 |
| `lip_smack` | 0 | 1306 | 0 | 0 | 99 | 0 | 0 | 1405 |
| `dishes` | 0 | 0 | 1301 | 0 | 0 | 0 | 0 | 1301 |
| `gasp` | 578 | 0 | 622 | 0 | 0 | 0 | 0 | 1200 |
| `cheer` | 0 | 0 | 1169 | 0 | 0 | 0 | 0 | 1169 |
| `explosion` | 0 | 0 | 1062 | 0 | 0 | 0 | 0 | 1062 |
| `chew` | 0 | 0 | 786 | 0 | 0 | 0 | 0 | 786 |
| `whisper` | 0 | 0 | 776 | 0 | 0 | 0 | 0 | 776 |
| `snore` | 91 | 6 | 632 | 8 | 0 | 0 | 0 | 737 |
| `scream` | 0 | 0 | 601 | 0 | 51 | 0 | 0 | 652 |
| `groan` | 0 | 0 | 496 | 69 | 45 | 0 | 0 | 610 |
| `glass_break` | 0 | 0 | 569 | 0 | 0 | 0 | 0 | 569 |
| `crowd` | 0 | 0 | 534 | 0 | 0 | 0 | 0 | 534 |
| `fan` | 0 | 468 | 40 | 0 | 0 | 0 | 0 | 508 |
| `buzzer` | 0 | 44 | 422 | 0 | 0 | 0 | 0 | 466 |
| `rain` | 0 | 0 | 465 | 0 | 0 | 0 | 0 | 465 |
| `sneeze` | 0 | 85 | 250 | 13 | 52 | 0 | 0 | 400 |
| `burp` | 0 | 0 | 387 | 0 | 0 | 0 | 0 | 387 |
| `door` | 0 | 0 | 385 | 0 | 0 | 0 | 0 | 385 |
| `yawn` | 269 | 31 | 15 | 0 | 48 | 0 | 0 | 363 |
| `phone_vibrate` | 0 | 219 | 144 | 0 | 0 | 0 | 0 | 363 |
| `hiccup` | 0 | 0 | 327 | 0 | 0 | 0 | 0 | 327 |
| `thunder` | 0 | 0 | 324 | 0 | 0 | 0 | 0 | 324 |
| `knock` | 0 | 0 | 182 | 0 | 0 | 0 | 0 | 182 |
| `nose_blow` | 0 | 93 | 0 | 0 | 48 | 0 | 0 | 141 |

## Feature pools (log-mel, fp16)

Fixed pools of 8 s examples are generated once, with augmentation, and cached as log-mel features: **171,000 training examples, 5,343 steps per epoch at batch size 32.** Synthetic mixes are stored twice (clean and one augmented copy), and real windows three times (clean and two augmented copies).

| feature pool | examples | events | without events | GB |
|---|---|---|---|---|
| `train_real_audioset` | 36,000 | 142,569 | 2,007 | 3.69 |
| `train_real_nv` | 27,000 | 26,110 | 7,807 | 2.77 |
| `train_real_nvtts` | 21,000 | 21,749 | 6,626 | 2.15 |
| `train_real_vaani` | 27,000 | 65,179 | 258 | 2.77 |
| `train_synth` | 60,000 | 83,344 | 9,582 | 6.15 |
| `val_noisy` | 600 | 1,429 | 65 | 0.06 |
| `val_real_audioset` | 800 | 3,729 | 32 | 0.08 |
| `val_real_nv` | 600 | 584 | 173 | 0.06 |
| `val_real_nvtts` | 400 | 558 | 46 | 0.04 |
| `val_real_vaani` | 800 | 1,777 | 15 | 0.08 |
| `val_synth` | 1,000 | 1,385 | 162 | 0.10 |
