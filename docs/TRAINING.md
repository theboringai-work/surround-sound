# Training

Two notebooks run the same pipeline:
- `training/train_local.ipynb` downloads the datasets to a local disk.
- `training/train_colab.ipynb` streams the Hugging Face datasets on Google Colab.

Both are **generated** by `training/build_notebooks.py`. Edit the builder, never the notebooks, and regenerate:

```bash
python training/build_notebooks.py        # rewrites training/train_local.ipynb and training/train_colab.ipynb
```

## Requirements

| | local | Colab |
|---|---|---|
| GPU | one 16 GB GPU (developed on a T4) | T4 runtime |
| disk | ~100 GB under `SED_WORKDIR` (downloads ~50 GB, audio banks ~32 GB, features ~18 GB) | ~35 GB of runtime disk; checkpoints backed up to Drive |
| RAM | ~11 GB peak for the notebook process | ~4–5 GB (smaller pools) |
| packages | `pip install -r requirements-train.txt` | installed by the first cell |
| access | `HF_TOKEN` with the gated [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) terms accepted | the same token, as a Colab secret |

## Running locally

```bash
export SED_WORKDIR=/path/to/fast/disk/sed-work     # default: <repo>/workdir
export HF_TOKEN=hf_...                              # optional: the notebook asks once if Vaani needs it
jupyter lab training/train_local.ipynb              # open from the repo root or from training/
```

1. In the setup cell, set `SMOKE = True` and run every cell. A tiny subset goes through every step in minutes. Fix any problem here first.
2. Set `SMOKE = False` and run all cells again. Smoke and full runs keep separate caches (`cache/smoke`, `cache/full`), and every step caches its output, so a re-run skips finished steps.
3. The model is exported to **`checkpoints_new/`**. `checkpoints/` is never overwritten. Compare the two with `current_vs_new.csv` (written when `checkpoints/` holds a model) and your own audio. Copy the new model over `checkpoints/` by hand once it's better.

Shared GPU: the setup cell prints the free GPU memory. Lower `CFG["bs"]` if training runs out of memory.

## Running on Colab

1. Runtime → Change runtime type → **T4 GPU**.
2. Accept the Vaani terms and add `HF_TOKEN` under the key icon in the left bar (Secrets), with notebook access turned on.
3. Optional: upload this repo's `checkpoints/` to `MyDrive/event-noise/checkpoints/`. That enables the warm start, speech-bed cleaning and the old-vs-new table.
4. Run with `SMOKE = True`, then `SMOKE = False` (Runtime → Restart session in between). Checkpoints are copied to Drive every epoch, and a disconnected run resumes when all cells are run again.

## Pipeline (cell by cell)

| step | what it does | output |
|---|---|---|
| 0 Setup | paths, `CFG` (sizes for smoke/full), GPU check | |
| 0b Download | local: NV38k, Vaani, NonverbalTTS parquet; OpenSLR 99; DNC zip; DEMAND channels 1 and 9. Each dataset gets a `.complete` marker. AudioSet is always streamed | `workdir/data/` |
| 1 Tags | `NV_MAP`, `VAANI_RULES`, `TTS_EMOJI`, `SLR_MAP`, `AS_RULES`, `DNC_MAP`, `DEMAND_MAP`; the 55-event list and which are speaker sounds | |
| 1b Model | log-mel → CNN → Transformer (the same code as `sed/model.py`) | |
| 2 Index | reads only the label and timestamp columns and records each clip's parquet row group; class list (≥ 80 events, 54 classes); splits | `cache/index/*.parquet` |
| 3 Banks | greedy row-group selection to fill per-class quotas; decode, resample to 16 kHz, int16 memory-mapped stores: event crops, speech beds, real windows | `cache/full/audio/` |
| 3a AudioSet | streamed row groups → real windows (10% event-free) and crops; validation = official evaluation split | |
| 3b NonverbalTTS | MMS_FA forced alignment → event timestamps (see `docs/DATA.md`) | |
| 3c OpenSLR 99 | energy-based trimming of each isolated vocalization | |
| 3d DNC/DEMAND | labelled recordings → ambient crops (≤ 16 s); the rest → 10 s background-noise chunks | |
| 3e Cleaning | the model in `checkpoints/` (if any) removes stretches with detected sounds (±0.3 s) from speech beds (speaker classes) and the noise pool (all classes) | `beds_clean.json`, `noise_clean.json` |
| 4 Examples | `MixDataset` (bed + 0–3 crops), `RealWindows` (real clips joined into 8 s windows), `Augmented` (see below); debug WAVs | `cache/full/debug/` |
| 5 Features | fixed pools → log-mel fp16 memmaps; validation sets per source + a noisy set | `cache/full/feats/` |
| 6 Model | warm start from `checkpoints/best.pt`: all weights except the output layer, plus the output rows of shared classes; new classes start fresh | |
| 7 Training | AdamW, OneCycle, AMP, gradient clipping at norm 5; checkpoint chosen by macro AP on real validation sets; resumes from `last.pt` | `cache/full/ckpt/` |
| 7b Thresholds | per class, 2-fold CV; balanced (F1) and sensitive (F2) profiles | `thresholds*.json`, `threshold_report.csv` |
| 8 Inference | `Detector`; ground truth vs detection on one held-out clip per source | |
| 9 Evaluation | event-level recall/precision/F1 on whole held-out clips; the model in `checkpoints/` is scored on the same clips | `event_eval.csv`, `current_vs_new.csv` |
| 10 Export | `best.pt`, `model.safetensors` + `config.json`, thresholds, reports, `train_info.json` | `checkpoints_new/` |
| 11 Whisper | rich transcript of an audio file | |

## Configuration (full run, local; as used for Experiment 1)

| key | value | meaning |
|---|---|---|
| `bank_cap` | 3000 | event crops per class per source (train split; validation gets 1/8) |
| `bed_cap` | 8000 | NV38k speech beds |
| `real_cap_vaani` / `real_cap_nv` / `real_cap_as` | 20000 / 15000 / 30000 | whole real clips kept |
| `as_rg_budget` / `as_val_rg` | 500 / 60 | AudioSet row groups streamed (72 clips each) |
| `pool_synth` | 30000 | synthetic mixes (×2 with augmentation) |
| `pool_real` | nv 9000, vaani 9000, nvtts 7000, audioset 12000 | real windows (×3 with augmentation) |
| `val_real` / `val_noisy` / `val_synth` | 600/800/400/800, 600, 1000 | validation windows |
| `epochs`, `bs`, `lr` | 10, 32, 3e-4 (5e-4 without warm start) | OneCycle with 10% warm-up, weight decay 0.01 |

The Colab copy uses smaller budgets: see the `CFG` cell of `train_colab.ipynb`.

## Augmentation (stored copies)

Each augmented copy draws independently:

| augmentation | probability | parameters |
|---|---|---|
| silence inserted outside events, labels shifted | 0.5 | 1–2 pauses of 0.2–1.5 s |
| reverb (synthetic exponential-decay room) | 0.25 | RT60 0.15–0.8 s |
| phone band + optional μ-law | 0.2 | 4th-order Butterworth 300–3400 Hz; μ-law 8-bit with p = 0.5 |
| background noise | 0.85 | real DNC/DEMAND noise 40% (SNR 3–25 dB), babble of 3–6 talkers 20% (12–30 dB), mains hum 10% (10–35 dB), white/pink/brown 30% (8–35 dB) |
| gain | 1 | −12 to +6 dB |

Synthetic mixes: speech bed at reference level, 0–3 events per window (none with p = 0.15, no speech with p = 0.10). Events are drawn uniformly over classes, placed at random times, at a speech-to-event ratio of −3 to 12 dB (10–25 dB below the reference level when there's no speech), with ±10% speed jitter (p = 0.3). Overall level −10 to +6 dB, plus a small noise floor. During training: SpecAugment (one time mask of 1–29 frames and one frequency mask of 1–7 mel bins, p = 0.5) and a random circular time shift.

## Loss

Frame targets are dilated by one frame (40 ms) on each side. The loss is

`BCE(frame logits, frame targets; pos_weight) + 0.25 · BCE(max over time of the logits, clip targets)`

with per-class `pos_weight = (negative frames / positive frames)`, clipped to [1, 100].

## Outputs of the Experiment 1 run

`checkpoints/` holds the Experiment 1 export: `training_log.csv` (per-epoch loss and validation AP), `threshold_report.csv`, `event_eval.csv` and a corrected `train_info.json`. The original export listed only four data sources; the export cell is fixed in the builder. Experiment 1 trained for 10 epochs at ~1,095 s per epoch on one T4, warm-started from a preliminary 31-class checkpoint of the same architecture (not released). The selected checkpoint is from epoch 6.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `TypeError: unsupported operand type(s) for +=: 'int' and 'list'` in Step 3a | fixed: `take_real` must be a `bool` |
| `LibsndfileError: Format not recognised` | handled: some Vaani rows hold HTML instead of audio; they're skipped and counted |
| CUDA out of memory | GPU shared with other jobs: lower `CFG["bs"]` |
| Vaani skipped, background classes missing | `HF_TOKEN` not set, or the Vaani terms not accepted |
| disk full | point `SED_WORKDIR` at a bigger disk, or lower the pool sizes in `CFG` |
