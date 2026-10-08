# Evaluation

All evaluation uses **real held-out recordings** that were never trained on:
- **NonVerbalSpeech-38K:** a 10% split by a stable hash of the clip.
- **Vaani:** verified clips from the 14 held-out districts.
- **NonverbalTTS:** official dev + test.
- **AudioSet-Strong:** the official evaluation split.

Each class is only scored on the datasets that label it. NonVerbalSpeech-38K, for example, has no labels for horns, so a horn predicted in one of its clips isn't counted.

## 1. Model selection: frame-level average precision

After every epoch, the model predicts every 40 ms frame of fixed validation windows:

| set | windows (8 s) |
|---|---|
| `real_nv` (NonVerbalSpeech-38K) | 600 |
| `real_vaani` | 800 |
| `real_nvtts` | 400 |
| `real_audioset` | 800 |
| `noisy` (augmented real windows, robustness check) | 600 |
| `synth` (synthetic mixes; reported, not used for selection) | 1,000 |

For each set, the macro average of per-class average precision is computed. The checkpoint with the best mean over the five real sets is kept: epoch 6 of 0–9, score 0.287. AP doesn't depend on a threshold, so selection rewards ranking real events above everything else. The per-epoch log is in `checkpoints/training_log.csv`.

## 2. Thresholds: 2-fold cross-validation

For each class, the validation windows of the sources that label it are split into two halves (even and odd windows). The threshold that maximizes frame F1 (balanced) or F2 (sensitive) on one half is scored on the other half, then the halves swap. The reported CV numbers are the mean of the two. The final thresholds are tuned on all windows.

A class is **weak** if it has fewer than 25 positive frames (1 s) or a CV F1 below 0.1. Weak classes are floored at 0.9 (balanced) or clamped to 0.6–0.8 (sensitive). `checkpoints/threshold_report.csv` has, per class: positive frames, AP, both thresholds, CV F1/F2, precision and recall.

## 3. Event level (`checkpoints/event_eval.csv`)

Whole held-out clips are run through the full detector, including windowing, smoothing, hysteresis and gap joining. That's up to 1,500 clips per source. Then:
- a true event is **found** if a prediction of the same class overlaps it;
- a prediction is **correct** if it overlaps a true event of the same class.

Recall, precision and F1 are computed per class, with both threshold profiles.

**Caveats:**
1. The thresholds were tuned on held-out data that overlaps these clips, so absolute numbers are somewhat optimistic.
2. The overlap criterion is lenient on boundaries; frame AP and CV F1 measure timing.
3. Unlabelled sounds in real clips count as false tags.
4. It's a single training run, so there are no confidence intervals.

### Macro summary (classes with ≥ 10 held-out events)
| classes (≥10 held-out events each) | classes | events | recall (sensitive) | recall (balanced) | precision (sensitive) | precision (balanced) | F1 (sensitive) | F1 (balanced) |
|---|---|---|---|---|---|---|---|---|
| speaker sounds | 16 | 3,271 | 0.46 | 0.33 | 0.38 | 0.40 | 0.38 | 0.34 |
| background sounds | 32 | 6,096 | 0.51 | 0.39 | 0.26 | 0.34 | 0.29 | 0.33 |
| all | 48 | 9,367 | 0.49 | 0.37 | 0.30 | 0.36 | 0.32 | 0.33 |

### Per class (S = sensitive profile, B = balanced profile)
| class | events | recall S | precision S | F1 S | recall B | precision B | F1 B |
|---|---|---|---|---|---|---|---|
| `music` | 1,361 | 0.84 | 0.76 | 0.80 | 0.75 | 0.85 | 0.80 |
| `breath` | 1,116 | 0.80 | 0.49 | 0.61 | 0.67 | 0.64 | 0.65 |
| `bird` | 791 | 0.64 | 0.60 | 0.62 | 0.56 | 0.80 | 0.66 |
| `laugh` | 587 | 0.79 | 0.52 | 0.63 | 0.57 | 0.77 | 0.66 |
| `vehicle` | 549 | 0.87 | 0.32 | 0.47 | 0.80 | 0.44 | 0.57 |
| `animal` | 538 | 0.52 | 0.12 | 0.19 | 0.25 | 0.25 | 0.25 |
| `sigh` | 389 | 0.84 | 0.72 | 0.78 | 0.60 | 0.86 | 0.70 |
| `cough` | 329 | 0.56 | 0.44 | 0.49 | 0.39 | 0.59 | 0.47 |
| `sniff` | 272 | 0.65 | 0.60 | 0.62 | 0.56 | 0.71 | 0.63 |
| `horn` | 258 | 0.76 | 0.71 | 0.74 | 0.64 | 0.83 | 0.72 |
| `construction` | 220 | 0.67 | 0.14 | 0.23 | 0.42 | 0.22 | 0.29 |
| `footsteps` | 218 | 0.50 | 0.10 | 0.17 | 0.30 | 0.29 | 0.30 |
| `wind` | 206 | 0.68 | 0.17 | 0.27 | 0.45 | 0.26 | 0.33 |
| `applause` | 199 | 0.34 | 0.47 | 0.40 | 0.27 | 0.47 | 0.34 |
| `water` | 192 | 0.75 | 0.19 | 0.30 | 0.49 | 0.27 | 0.34 |
| `throat_clear` | 190 | 0.62 | 0.38 | 0.47 | 0.45 | 0.49 | 0.47 |
| `child_voice` | 179 | 0.48 | 0.22 | 0.31 | 0.25 | 0.39 | 0.30 |
| `beep` | 165 | 0.56 | 0.26 | 0.35 | 0.52 | 0.27 | 0.35 |
| `machine` | 137 | 0.51 | 0.12 | 0.20 | 0.45 | 0.12 | 0.18 |
| `dog_bark` | 128 | 0.66 | 0.17 | 0.27 | 0.45 | 0.77 | 0.57 |
| `siren` | 114 | 0.58 | 0.24 | 0.34 | 0.50 | 0.29 | 0.36 |
| `shout` | 97 | 0.45 | 0.19 | 0.27 | 0.35 | 0.19 | 0.24 |
| `insect` | 87 | 0.42 | 0.23 | 0.30 | 0.39 | 0.35 | 0.37 |
| `bell` | 87 | 0.81 | 0.19 | 0.30 | 0.41 | 0.36 | 0.38 |
| `whistle` | 83 | 0.28 | 0.38 | 0.32 | 0.17 | 0.44 | 0.24 |
| `gunshot` | 75 | 0.81 | 0.22 | 0.34 | 0.79 | 0.37 | 0.50 |
| `cheer` | 73 | 0.26 | 0.39 | 0.31 | 0.23 | 0.43 | 0.30 |
| `phone_ring` | 64 | 0.70 | 0.24 | 0.36 | 0.67 | 0.23 | 0.35 |
| `cry` | 61 | 0.43 | 0.11 | 0.17 | 0.08 | 0.09 | 0.09 |
| `glass_break` | 58 | 0.12 | 1.00 | 0.21 | 0.12 | 1.00 | 0.21 |
| `explosion` | 54 | 0.63 | 0.16 | 0.25 | 0.63 | 0.16 | 0.25 |
| `dishes` | 50 | 0.42 | 0.11 | 0.17 | 0.38 | 0.09 | 0.15 |
| `rain` | 43 | 0.12 | 0.07 | 0.08 | 0.12 | 0.07 | 0.08 |
| `gasp` | 41 | 0.37 | 0.12 | 0.19 | 0.34 | 0.13 | 0.18 |
| `crowd` | 41 | 0.61 | 0.17 | 0.27 | 0.41 | 0.34 | 0.37 |
| `whisper` | 36 | 0.36 | 0.83 | 0.50 | 0.28 | 0.75 | 0.41 |
| `groan` | 36 | 0.19 | 0.09 | 0.12 | 0.11 | 0.14 | 0.12 |
| `scream` | 32 | 0.16 | 0.50 | 0.24 | 0.16 | 0.50 | 0.24 |
| `baby_cry` | 31 | 0.39 | 0.23 | 0.29 | 0.29 | 0.21 | 0.25 |
| `chew` | 29 | 0.10 | 0.08 | 0.09 | 0.00 | 0.00 | 0.00 |
| `snore` | 28 | 0.68 | 0.43 | 0.53 | 0.68 | 0.46 | 0.55 |
| `buzzer` | 25 | 0.20 | 0.06 | 0.09 | 0.16 | 0.17 | 0.16 |
| `tv_speaker` | 23 | 0.43 | 0.06 | 0.10 | 0.22 | 0.09 | 0.12 |
| `door` | 21 | 0.05 | 0.08 | 0.06 | 0.00 | 0.00 | 0.00 |
| `yawn` | 14 | 0.07 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 |
| `hiccup` | 14 | 0.21 | 0.60 | 0.32 | 0.00 | 0.00 | 0.00 |
| `fan` | 13 | 0.15 | 0.03 | 0.06 | 0.08 | 0.03 | 0.04 |
| `thunder` | 13 | 0.54 | 0.17 | 0.26 | 0.31 | 0.18 | 0.23 |
| `lip_smack` | 9 | 0.22 | 0.15 | 0.18 | 0.22 | 0.25 | 0.23 |
| `sneeze` | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `burp` | 7 | 0.57 | 0.16 | 0.25 | 0.29 | 0.10 | 0.15 |
| `knock` | 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `phone_vibrate` | 4 | 0.25 | 0.09 | 0.13 | 0.25 | 0.09 | 0.13 |
| `nose_blow` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
