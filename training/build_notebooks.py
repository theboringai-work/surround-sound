"""Builds training/train_local.ipynb and training/train_colab.ipynb from one template.

Edit this file, not the notebooks, then run `python training/build_notebooks.py` so both copies stay in sync.
"""
import json, os, sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))     # the notebooks are written next to this script
CELLS = []  # (variant, kind, source); variant in {"both", "local", "colab"}


def md(src, v="both"): CELLS.append((v, "markdown", src.strip("\n")))
def code(src, v="both"): CELLS.append((v, "code", src.strip("\n")))


# ----------------------------------------------------------------------------------------------- intro
md(r'''
# Event / noise detection with timestamps: local training (one GPU, data on a local disk)

This is the **local** copy. `train_colab.ipynb` is the Colab copy of the same pipeline. Only the setup cell, the data access and the sizes differ.

**Data.** Everything is downloaded once to `SED_WORKDIR` (default: `<repo>/workdir`; point it at a large, fast local disk, ~100 GB for a full run) and every step reads it from there.

| dataset | what it adds |
|---|---|
| `nonverbalspeech/nonverbalspeech38k` | 38.7k Mandarin/English clips, one timestamped vocal event each: event crops, speech beds, and **whole clips as real annotated windows** |
| `ARTPARK-IISc/Vaani-Noise-Event-Dataset` (gated) | 90.6k Indic phone recordings, 319 free-text noise tags with timestamps |
| `deepvk/NonverbalTTS` | 6.3k **English** clips (VoxCeleb, Expresso). Its labels are emoji inside the transcript and have no timestamps. A forced aligner (torchaudio MMS_FA) times the words, and each sound is placed in the pause where its emoji sits |
| `enyoukai/AudioSet-Strong` (streamed, 121 GB) | AudioSet (the data PANNs were trained on) with **timestamped** labels: 79.5k + 14k 10 s clips. The row groups that best cover our classes become real windows and crops; brings breath, laugh, cough, sneeze, siren, fan, phone sounds, scream… in many acoustic domains |
| OpenSLR 99 (Deeply Nonverbal Vocalization) | ~730 isolated phone recordings in 16 classes (cough, sneeze, nose-blow, yawn, lip-smack, moan…) |
| DNC: Dataset for Noise Classification (TU Berlin) | 4.4k home background recordings: TV/radio → `tv_speaker`, cars → `vehicle`, music → `music`, appliances → `machine`; quiet rooms and people go into the background-noise pool |
| DEMAND (Zenodo 1227121) | 18 real environments, 5 min each: traffic/bus/car/metro → `vehicle`, washing room → `machine`; kitchen, office, café, park etc. go into the background-noise pool |

**Design choices:**
1. **Real annotated audio.** Whole NV clips, Vaani clips, aligned NonverbalTTS clips and AudioSet-Strong clips are real training windows, alongside synthetic mixtures.
2. **Augmented copies.** Every training window is stored 2–3× (`aug_copies_*`): clean, plus copies with background noise (real DNC/DEMAND recordings, white, pink, brown, mains hum, babble), inserted silences, reverb, phone-band filtering and gain changes. The labels shift with the inserted silence.
3. **Cleaner speech beds.** If a model is present in `checkpoints/`, it removes stretches of the speech beds that contain unlabelled breaths or laughs, which would otherwise teach the model to ignore real events.
4. **Model selection by average precision on real held-out audio** (per dataset), not on synthetic data.
5. **Warm start** from `checkpoints/best.pt` if present, with head rows copied for classes that already exist.
6. **Event-level evaluation** on whole held-out clips at the end; if a model is present in `checkpoints/`, it is scored on the same clips for comparison.

Run `SMOKE = True` first. It covers every step in about 10–15 minutes. Then set `SMOKE = False`. Every step caches its output and is skipped on re-run.
''', "local")

md(r'''
# Event / noise detection with timestamps: Colab training (T4, 12 GB RAM)

This is the **Colab** copy. `train_local.ipynb` is the copy for the local machine, which downloads everything to its NVMe volume.

**Data.** The four Hugging Face datasets (including AudioSet-Strong, 121 GB) are not downloaded. Their parquet row groups are streamed into memory, decoded, resampled to 16 kHz and written as int16 to memory-mapped files on the runtime's local disk. OpenSLR 99, DNC and DEMAND are downloaded: DNC is read straight from its zip, and only mic channels 1 and 9 of DEMAND are kept.

| dataset | what it adds |
|---|---|
| `nonverbalspeech/nonverbalspeech38k` | 38.7k Mandarin/English clips, one timestamped vocal event each: event crops, speech beds, and **whole clips as real annotated windows** |
| `ARTPARK-IISc/Vaani-Noise-Event-Dataset` (gated) | 90.6k Indic phone recordings, 319 free-text noise tags with timestamps |
| `deepvk/NonverbalTTS` | 6.3k **English** clips (VoxCeleb, Expresso). Its labels are emoji inside the transcript and have no timestamps. A forced aligner (torchaudio MMS_FA) times the words, and each sound is placed in the pause where its emoji sits |
| `enyoukai/AudioSet-Strong` (streamed, 121 GB) | AudioSet (the data PANNs were trained on) with **timestamped** labels: 79.5k + 14k 10 s clips. The row groups that best cover our classes become real windows and crops; brings breath, laugh, cough, sneeze, siren, fan, phone sounds, scream… in many acoustic domains |
| OpenSLR 99 (Deeply Nonverbal Vocalization) | ~730 isolated phone recordings in 16 classes (cough, sneeze, nose-blow, yawn, lip-smack, moan…) |
| DNC: Dataset for Noise Classification (TU Berlin) | 4.4k home background recordings: TV/radio → `tv_speaker`, cars → `vehicle`, music → `music`, appliances → `machine`; quiet rooms and people go into the background-noise pool |
| DEMAND (Zenodo 1227121) | 18 real environments, 5 min each: traffic/bus/car/metro → `vehicle`, washing room → `machine`; kitchen, office, café, park etc. go into the background-noise pool |

**Design choices:**
1. **Real annotated audio.** Whole NV clips, Vaani clips, aligned NonverbalTTS clips and AudioSet-Strong clips are real training windows, alongside synthetic mixtures.
2. **Augmented copies.** Every training window is stored 2–3× (`aug_copies_*`): clean, plus copies with background noise (real DNC/DEMAND recordings, white, pink, brown, mains hum, babble), inserted silences, reverb, phone-band filtering and gain changes.
3. **Cleaner speech beds.** A model in `checkpoints/` (uploaded as in step 3 below) removes bed stretches that contain unlabelled vocal sounds.
4. **Model selection by average precision on real held-out audio**, per dataset.
5. **Warm start** from that model's `best.pt` if it's uploaded.
6. **Event-level evaluation** at the end, with the uploaded model scored on the same clips for comparison.

**Setup**
1. Runtime → Change runtime type → **T4 GPU**. On the free tier, connect only when you're ready to start.
2. Accept the terms of the gated [Vaani dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) on Hugging Face. Then add `HF_TOKEN` under the key icon in the left bar (Secrets) and turn on notebook access.
3. Optional, recommended: in Google Drive, create `MyDrive/event-noise/checkpoints/` and upload the repo's `checkpoints/` folder into it (`best.pt`, `thresholds*.json`, `detect_config.json`, ~30 MB). That enables the warm start, bed/noise cleaning and the comparison table.
4. Run the cells in order with `SMOKE = True` (about 15–20 min, ~7 GB of disk). Then set `SMOKE = False`, choose *Runtime → Restart session*, and run all the cells again.

**Disk (full run, default sizes).** About 35 GB on the runtime's local disk. A T4 runtime normally has much more than that free; the setup cell prints it.

| what | GB |
|---|---|
| downloads: DNC zip 4.2, DEMAND channels 0.4 (zips deleted one by one), OpenSLR 99 0.1 | 4.7 |
| model weights: MMS aligner 1.2, Whisper small 0.5 | 1.7 |
| audio banks (int16, 16 kHz): real windows ~6.6 + AudioSet ~3, speech beds ~1.6, event crops ~2.3, noise pool ~0.7 | ~14 |
| feature cache (fp16 log-mel, ~120k examples × 100 KB) | ~12.3 |
| checkpoints, index, debug audio | 0.2 |
| **total** | **~33–35** |

Peak RAM is about 4–5 GB of the 12.7 GB. **Google Drive:** checkpoints are backed up every epoch, about 0.2 GB in all. With `BACKUP_CACHE = True`, the banks and features are backed up too (~26 GB of Drive space), which saves about 2 h if the runtime resets.

**Time (full run, unshared T4):**
- index ~5 min
- streaming and banks ~40 min, plus AudioSet streaming (150 row groups, ~13 GB read, ~3 GB kept) ~20 min
- NonverbalTTS alignment ~35 min
- DNC/DEMAND download ~10–20 min
- features ~30 min
- training 10 epochs ~2.5–3 h
- tuning and evaluation ~20 min

That's about **5–6 h** in total. Training resumes from Drive after a disconnect: re-run all the cells, and finished steps are skipped or restored.
''', "colab")

# ----------------------------------------------------------------------------------------------- setup
md("## Step 0: Setup")

code(r'''
# Local copy: datasets, caches and checkpoints live under SED_WORKDIR (default <repo>/workdir). Put it on a
# large, fast local disk (~100 GB for a full run). The Hugging Face and torch caches go there too, so set
# this before huggingface_hub / torchaudio load. Run the notebook from the repo root or from training/.
import os, sys
REPO = next((p for p in (os.getcwd(), os.path.dirname(os.getcwd())) if os.path.isdir(os.path.join(p, "sed"))), None)
assert REPO, "open this notebook from the repository (the folder with sed/) or its training/ folder"
ROOT = os.path.abspath(os.environ.get("SED_WORKDIR", os.path.join(REPO, "workdir")))
os.environ.setdefault("HF_HOME", f"{ROOT}/hf_home")
os.environ.setdefault("TORCH_HOME", f"{ROOT}/torch_home")

LOCAL = True
SMOKE = True          # True: tiny subset, runs every step end to end. False: full run.

DATA  = f"{ROOT}/data"                       # downloaded datasets
CACHE = f"{ROOT}/cache"                      # index, audio banks, features, checkpoints
OLD_DIR   = f"{REPO}/checkpoints"            # current model: warm start, bed cleaning, comparison table
INIT_FROM = f"{OLD_DIR}/best.pt"             # set to None to train from scratch
EXPORT    = f"{REPO}/checkpoints_new"        # the finished model goes here (checkpoints/ is never overwritten)
BACKUP, BACKUP_CACHE = None, False           # (Colab copy: Google Drive backup)

CFG = dict(
    nv_rg_budget   = 3   if SMOKE else 10**6,  # NV38k row groups to read (10**6 = whatever fills the quotas)
    va_rg_budget   = 4   if SMOKE else 10**6,  # Vaani row groups
    va_val_rg      = 2   if SMOKE else 10**6,  # + Vaani row groups richest in held-out (verified) val clips
    bank_cap       = 8   if SMOKE else 3000,   # event crops per class per dataset (train split)
    bed_cap        = 40  if SMOKE else 8000,   # speech-bed segments from NV non-event regions
    real_cap_vaani = 80  if SMOKE else 20000,  # whole Vaani clips with timestamps
    real_cap_nv    = 60  if SMOKE else 15000,  # whole NV38k clips with their timestamped event
    tts_cap        = dict(train=40, val=20) if SMOKE else None,   # NonverbalTTS clips to align (None = all)
    as_rg_budget   = 3   if SMOKE else 500,    # AudioSet-Strong row groups to stream (72 clips, ~89 MB each)
    as_val_rg      = 1   if SMOKE else 60,     # + row groups from its test split (held-out real windows)
    real_cap_as    = 80  if SMOKE else 30000,  # whole AudioSet clips (10 s) kept as real windows (~0.3 MB each)
    slr_cap        = 8   if SMOKE else None,   # OpenSLR 99 clips per class (None = all)
    noise_cap      = 20  if SMOKE else None,   # DNC/DEMAND background-noise chunks per source (None = all)
    pool_synth     = 128 if SMOKE else 30000,  # synthetic mixes (before augmentation copies)
    pool_real      = dict(nv=32, vaani=32, nvtts=32, audioset=32) if SMOKE else dict(nv=9000, vaani=9000, nvtts=7000, audioset=12000),
    aug_copies_synth = 1,                      # +1 augmented copy of every synthetic mix -> 2x
    aug_copies_real  = 2,                      # +2 augmented copies of every real window  -> 3x
    val_synth      = 32  if SMOKE else 1000,
    val_real       = dict(nv=16, vaani=16, nvtts=16, audioset=16) if SMOKE else dict(nv=600, vaani=800, nvtts=400, audioset=800),
    val_noisy      = 16  if SMOKE else 600,    # augmented real val windows: robustness check
    eval_clips     = 30  if SMOKE else 1500,   # whole held-out clips per dataset for the event-level table
    epochs         = 1   if SMOKE else 10,
    bs             = 16  if SMOKE else 32,
    workers        = max(2, min(12, (os.cpu_count() or 4) - 2)),   # DataLoader workers
    decode_threads = max(2, min(12, (os.cpu_count() or 4) - 2)),
)
''', "local")

code(r'''
# Colab already has torch/torchaudio/pyarrow/huggingface_hub/soundfile; this only makes sure they're recent enough.
!pip -q install -U "huggingface_hub>=0.23" soundfile
''', "colab")

code(r'''
import os, sys
LOCAL = False
SMOKE = True          # True: tiny subset, runs every step end to end. False: full run.

if not os.environ.get("HF_TOKEN"):
    from google.colab import userdata            # key icon (left bar) -> Secrets -> add HF_TOKEN
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
assert os.environ.get("HF_TOKEN"), "Vaani is gated: accept its terms on the HF page and set HF_TOKEN"

# Everything is written to the runtime's local disk (fast; memory-mapped files on Drive would be very slow).
# Google Drive is used only as a backup, so a disconnected run can resume:
#   - USE_DRIVE: checkpoints (every epoch), the metadata index and the exported model are copied to Drive (~0.2 GB)
#   - BACKUP_CACHE: also back up the audio banks and features (~26 GB of Drive space; saves ~2 h after a reset)
USE_DRIVE    = True
BACKUP_CACHE = False
ROOT  = "/content/event-noise"
os.environ.setdefault("TORCH_HOME", f"{ROOT}/torch_home")     # MMS aligner weights (1.2 GB)
DATA  = f"{ROOT}/data"                       # OpenSLR 99, DNC, DEMAND (the HF datasets are streamed)
CACHE = f"{ROOT}/cache"
BACKUP = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BACKUP = "/content/drive/MyDrive/event-noise"
# The current model enables warm start, bed/noise cleaning and the comparison table. Upload the repo's checkpoints/
# folder to Drive as MyDrive/event-noise/checkpoints (or to /content/checkpoints without Drive). Optional.
OLD_DIR   = f"{BACKUP}/checkpoints" if BACKUP else "/content/checkpoints"
INIT_FROM = f"{OLD_DIR}/best.pt"             # warm start if present; set to None to train from scratch
EXPORT    = f"{BACKUP}/checkpoints_new" if BACKUP else f"{ROOT}/checkpoints_new"

CFG = dict(
    nv_rg_budget   = 3   if SMOKE else 90,     # NV38k row groups to stream (~55 MB each)
    va_rg_budget   = 4   if SMOKE else 250,    # Vaani row groups (~20 MB each)
    va_val_rg      = 2   if SMOKE else 30,     # + Vaani row groups richest in held-out (verified) val clips
    bank_cap       = 8   if SMOKE else 800,    # event crops per class per dataset (train split)
    bed_cap        = 40  if SMOKE else 4000,   # speech-bed segments from NV non-event regions
    real_cap_vaani = 80  if SMOKE else 15000,  # whole Vaani clips with timestamps
    real_cap_nv    = 60  if SMOKE else 6000,   # whole NV38k clips with their timestamped event
    tts_cap        = dict(train=40, val=20) if SMOKE else None,   # NonverbalTTS clips to align (None = all, 4.2 GB)
    as_rg_budget   = 3   if SMOKE else 150,    # AudioSet-Strong row groups to stream (72 clips, ~89 MB each)
    as_val_rg      = 1   if SMOKE else 20,
    real_cap_as    = 80  if SMOKE else 9000,
    slr_cap        = 8   if SMOKE else None,   # OpenSLR 99 clips per class (None = all)
    noise_cap      = 20  if SMOKE else None,   # DNC/DEMAND background-noise chunks per source (None = all)
    pool_synth     = 128 if SMOKE else 25000,  # synthetic mixes (before augmentation copies)
    pool_real      = dict(nv=32, vaani=32, nvtts=32, audioset=32) if SMOKE else dict(nv=5000, vaani=6000, nvtts=4000, audioset=5000),
    aug_copies_synth = 1,                      # +1 augmented copy of every synthetic mix -> 2x
    aug_copies_real  = 2,                      # +2 augmented copies of every real window  -> 3x
    val_synth      = 32  if SMOKE else 1000,
    val_real       = dict(nv=16, vaani=16, nvtts=16, audioset=16) if SMOKE else dict(nv=500, vaani=800, nvtts=400, audioset=600),
    val_noisy      = 16  if SMOKE else 500,
    eval_clips     = 30  if SMOKE else 1000,
    epochs         = 1   if SMOKE else 10,
    bs             = 16  if SMOKE else 32,
    workers        = 2,                        # Colab has 2 vCPUs
    decode_threads = 2,
)
import shutil
free_gb = shutil.disk_usage("/content").free / 1e9
print(f"local disk free: {free_gb:.0f} GB (the full run needs ~35 GB, smoke ~7 GB)")
if free_gb < (8 if SMOKE else 37): print("WARNING: not enough local disk; lower the pool sizes in CFG")
''', "colab")

code(r'''
import io, re, gc, json, math, time, zlib, glob, shutil, tarfile, zipfile, threading, itertools, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, psutil, scipy.signal
import torch, torchaudio, soundfile as sf
import torch.nn as nn, torch.nn.functional as F
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, snapshot_download
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm.auto import tqdm

RUN = os.path.join(CACHE, "smoke" if SMOKE else "full")     # banks / features / checkpoints per mode
for d in (DATA, CACHE, RUN): os.makedirs(d, exist_ok=True)
SR, N_MELS, FRAME, DUR = 16000, 64, 0.04, 8.0
NV_REPO, VA_REPO, TTS_REPO = "nonverbalspeech/nonverbalspeech38k", "ARTPARK-IISc/Vaani-Noise-Event-Dataset", "deepvk/NonverbalTTS"
AS_REPO = "enyoukai/AudioSet-Strong"          # AudioSet with temporally-strong labels (the data family PANNs were trained on)
PQ_GLOB = {NV_REPO: "data/*.parquet", VA_REPO: "data/*.parquet", TTS_REPO: "default/*/*.parquet", AS_REPO: "data/*.parquet"}
STREAMED = {AS_REPO}                           # always streamed from the Hub (121 GB; only the clips used are stored)
SLR_URL = "https://www.openslr.org/resources/99/NonverbalVocalization.tgz"
SRCS = ["nv", "vaani", "nvtts", "audioset"]   # sources of real annotated windows
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
CFG["lr"] = 3e-4 if INIT_FROM and os.path.exists(INIT_FROM) else 5e-4   # lower peak LR when fine-tuning
fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))

def sync_dir(src, dst):          # copy files that are new or changed (size or mtime) from src to dst
    for root, _, files in os.walk(src):
        for f in files:
            a = os.path.join(root, f); b = os.path.join(dst, os.path.relpath(a, src))
            if f.endswith(".part"): continue
            if not os.path.exists(b) or os.path.getsize(a) != os.path.getsize(b) or os.path.getmtime(a) > os.path.getmtime(b) + 1:
                os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy2(a, b)

def backup(path):               # mirror a cache folder to BACKUP (Colab: Google Drive); does nothing locally
    if BACKUP and os.path.exists(path): sync_dir(path, os.path.join(BACKUP, os.path.relpath(path, ROOT)))

if BACKUP and os.path.isdir(f"{BACKUP}/cache"):          # resume after a runtime reset
    sync_dir(f"{BACKUP}/cache", CACHE); print("restored cache from", f"{BACKUP}/cache")

def mem():
    vm = psutil.virtual_memory()
    g = f" | GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB peak" if torch.cuda.is_available() else ""
    return f"RAM: process {psutil.Process().memory_info().rss/1e9:.2f} GB, free {vm.available/1e9:.1f} GB{g}"

if dev == "cuda":
    free, total = torch.cuda.mem_get_info()
    print(f"GPU {torch.cuda.get_device_name()}: {free/1e9:.1f} of {total/1e9:.1f} GB free")
    if free < 4e9: print("warning: the GPU is shared with other jobs; lower CFG['bs'] if training runs out of memory")
print(RUN, dev, "| disk free:", f"{shutil.disk_usage(CACHE).free/1e9:.0f} GB |", mem())
''')

# ----------------------------------------------------------------------------------------------- download
md(r'''
## Step 0b: Download the datasets to the NVMe volume
About 50 GB in total: NV38k 21 GB, Vaani 17 GB, NonverbalTTS 4.2 GB, DNC 3.9 GB, DEMAND ~2 GB (16 kHz zips), OpenSLR 99 45 MB. Each finished dataset gets a `.complete` marker and is skipped on re-run. Vaani is gated. If `HF_TOKEN` isn't set when Vaani is needed, the cell asks for it once and keeps it only in this kernel's memory.
''', "local")

code(r'''
def fetch_hf(repo):
    out = f"{DATA}/{repo}"
    if not os.path.exists(f"{out}/.complete"):
        if repo == VA_REPO and not os.environ.get("HF_TOKEN"):
            from getpass import getpass
            os.environ["HF_TOKEN"] = getpass("HF token (accept the Vaani terms on its HF page first): ")
        snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=[PQ_GLOB[repo]],
                          token=os.environ.get("HF_TOKEN"), max_workers=16)
        open(f"{out}/.complete", "w").close()
    files = glob.glob(f"{out}/{PQ_GLOB[repo]}")
    print(f"{repo:42s} {len(files):4d} parquet files, {sum(map(os.path.getsize, files))/1e9:5.1f} GB")

def fetch_slr99():
    out = f"{DATA}/openslr99"; os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/.complete"):
        tgz = f"{out}/NonverbalVocalization.tgz"
        urllib.request.urlretrieve(SLR_URL, tgz + ".part"); os.replace(tgz + ".part", tgz)
        tarfile.open(tgz).extractall(out)
        open(f"{out}/.complete", "w").close()
    print(f"OpenSLR 99: {len(glob.glob(f'{out}/NonverbalVocalization/*/*.wav'))} wav files")

for repo in (TTS_REPO, NV_REPO, VA_REPO):
    fetch_hf(repo)
DNC_URL = "https://depositonce.tu-berlin.de/bitstream/11303/12845/2/DNC.zip"
DEMAND_URL = "https://zenodo.org/records/1227121/files/{}.zip?download=1"
DEMAND_ENVS = ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING", "OOFFICE",
               "PCAFETER", "PRESTO", "PSTATION", "SCAFE", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO"]

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)

def fetch_dnc():                    # DNC (4.2 GB zip of 48 kHz wavs, label = filename prefix); read straight from the zip
    out = f"{DATA}/dnc"
    if not os.path.exists(f"{out}/.complete"):
        download(DNC_URL, f"{out}/DNC.zip"); open(f"{out}/.complete", "w").close()
    with zipfile.ZipFile(f"{out}/DNC.zip") as z:
        print(f"DNC: {sum(n.lower().endswith('.wav') for n in z.namelist())} wav files")

def fetch_demand():                 # DEMAND: 16 kHz zips (SCAFE only at 48 kHz); only mic channels 1 and 9 are kept
    out = f"{DATA}/demand"
    if not os.path.exists(f"{out}/.complete"):
        for e in tqdm(DEMAND_ENVS, desc="DEMAND"):
            name = f"{e}_48k.zip" if e == "SCAFE" else f"{e}_16k.zip"
            if all(os.path.exists(f"{out}/{e}/{ch}.wav") for ch in ("ch01", "ch09")): continue
            download(DEMAND_URL.format(name[:-4]), f"{out}/{name}")
            with zipfile.ZipFile(f"{out}/{name}") as z:
                for ch in ("ch01", "ch09"): z.extract(f"{e}/{ch}.wav", out)
            if not LOCAL: os.remove(f"{out}/{name}")             # Colab: save disk
        open(f"{out}/.complete", "w").close()
    print(f"DEMAND: {len(glob.glob(f'{out}/*/ch01.wav'))} environments")

fetch_slr99()
fetch_dnc()
fetch_demand()
print("disk free:", f"{shutil.disk_usage(DATA).free/1e9:.0f} GB")
''', "local")

md(r'''
## Step 0b: Download the small datasets: OpenSLR 99 (45 MB), DNC (4.2 GB zip, read in place), DEMAND (2 GB of zips, 0.37 GB kept). The Hugging Face datasets are streamed.
''', "colab")

code(r'''
def fetch_slr99():
    out = f"{DATA}/openslr99"; os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/.complete"):
        tgz = f"{out}/NonverbalVocalization.tgz"
        urllib.request.urlretrieve(SLR_URL, tgz + ".part"); os.replace(tgz + ".part", tgz)
        tarfile.open(tgz).extractall(out)
        open(f"{out}/.complete", "w").close()
    print(f"OpenSLR 99: {len(glob.glob(f'{out}/NonverbalVocalization/*/*.wav'))} wav files")

DNC_URL = "https://depositonce.tu-berlin.de/bitstream/11303/12845/2/DNC.zip"
DEMAND_URL = "https://zenodo.org/records/1227121/files/{}.zip?download=1"
DEMAND_ENVS = ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING", "OOFFICE",
               "PCAFETER", "PRESTO", "PSTATION", "SCAFE", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO"]

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)

def fetch_dnc():                    # DNC (4.2 GB zip of 48 kHz wavs, label = filename prefix); read straight from the zip
    out = f"{DATA}/dnc"
    if not os.path.exists(f"{out}/.complete"):
        download(DNC_URL, f"{out}/DNC.zip"); open(f"{out}/.complete", "w").close()
    with zipfile.ZipFile(f"{out}/DNC.zip") as z:
        print(f"DNC: {sum(n.lower().endswith('.wav') for n in z.namelist())} wav files")

def fetch_demand():                 # DEMAND: 16 kHz zips (SCAFE only at 48 kHz); only mic channels 1 and 9 are kept
    out = f"{DATA}/demand"
    if not os.path.exists(f"{out}/.complete"):
        for e in tqdm(DEMAND_ENVS, desc="DEMAND"):
            name = f"{e}_48k.zip" if e == "SCAFE" else f"{e}_16k.zip"
            if all(os.path.exists(f"{out}/{e}/{ch}.wav") for ch in ("ch01", "ch09")): continue
            download(DEMAND_URL.format(name[:-4]), f"{out}/{name}")
            with zipfile.ZipFile(f"{out}/{name}") as z:
                for ch in ("ch01", "ch09"): z.extract(f"{e}/{ch}.wav", out)
            if not LOCAL: os.remove(f"{out}/{name}")             # Colab: save disk
        open(f"{out}/.complete", "w").close()
    print(f"DEMAND: {len(glob.glob(f'{out}/*/ch01.wav'))} environments")

fetch_slr99()
fetch_dnc()        # 4.2 GB zip, read in place
fetch_demand()     # ~2 GB of zips, one at a time; 0.37 GB kept
''', "colab")

code(r'''
# One access layer for both copies: local parquet files (LOCAL) or HTTP range reads from the Hub (Colab).
def pq_files(repo):            # parquet paths relative to the dataset root, e.g. "data/train-00000-of-00046.parquet"
    if LOCAL and repo not in STREAMED:
        base = f"{DATA}/{repo}"
        return sorted(os.path.relpath(p, base) for p in glob.glob(f"{base}/{PQ_GLOB[repo]}"))
    return sorted(p.split(f"datasets/{repo}/", 1)[1] for p in fs.glob(f"datasets/{repo}/{PQ_GLOB[repo]}"))

def pq_open(repo, rel, block=8 << 20):
    return open(f"{DATA}/{repo}/{rel}", "rb") if LOCAL and repo not in STREAMED else fs.open(f"datasets/{repo}/{rel}", block_size=block)

HAVE_VAANI = bool(pq_files(VA_REPO)) if LOCAL else bool(os.environ.get("HF_TOKEN"))
print({r: len(pq_files(r)) for r in (NV_REPO, VA_REPO, TTS_REPO, AS_REPO)}, "| Vaani available:", HAVE_VAANI)
''')

# ----------------------------------------------------------------------------------------------- taxonomy
md(r'''
## Step 1: Canonical tags
Every dataset's labels map to one canonical tag set:
- **NV38k**: direct map.
- **Vaani**: 319 free-text tags such as `<horn>`, `<honking>` and `<vehicle horn>` go through ordered regex rules. `None` means a speech tag, which is ignored. `DROP` means an odd tag; a clip containing one isn't used as real annotated data.
- **NonverbalTTS**: one emoji per sound (🌬️ breath, 🤣 laugh, 👃 sniff, 😷 cough, 🗣 throat clearing, 😤 sigh, 😖 groan, 🤧 sneeze, 😴 snore, 🐖 grunt). The mapping was matched to the paper's per-type counts. Grunt merges into `groan`.
- **OpenSLR 99**: folder names. Panting → breath, lip-popping → lip_smack, moaning → groan. Teeth sounds are dropped.
- **DNC / DEMAND**: whole recordings of one kind of background. Labelled kinds become ambient event crops. The rest (quiet rooms, people, kitchen, office, café…) becomes unlabelled background noise for augmentation.

Classes with fewer than `MIN_EVENTS` events across all datasets are dropped. Check the table that Step 2 prints.
''')

code(r'''
NV_MAP = {"sigh": "sigh", "sniff": "sniff", "laughing": "laugh", "coughing": "cough",
          "throatclearing": "throat_clear", "breath": "breath", "crying": "cry",
          "gasp": "gasp", "yawn": "yawn", "snore": "snore"}

VAANI_RULES = [   # first match wins, so order matters
    (r"\b(child|children|kids?|baby|human|people)\b.*\b(talking|speaking|voices?)\b|human (noise|sound)", None),
    (r"\b(child|children|kids?|baby)\b.*\bcry",                        "baby_cry"),
    (r"laugh",                                                         "laugh"),
    (r"cough",                                                         "cough"),
    (r"throat",                                                        "throat_clear"),
    (r"sneez",                                                         "sneeze"),
    (r"nose (blow|clear)",                                             "nose_blow"),
    (r"lip.?smack",                                                    "lip_smack"),
    (r"breath|inhal|exhal",                                            "breath"),
    (r"sniff|snort",                                                   "sniff"),
    (r"yawn|yawing",                                                   "yawn"),
    (r"snor",                                                          "snore"),
    (r"gasp",                                                          "gasp"),
    (r"\b(child|children|kids?|baby)\b",                               "child_voice"),   # yelling / playing / noise
    (r"siren|ambulance|alarm",                                         "siren"),
    (r"horn|honk",                                                     "horn"),
    (r"bark|\bdog\b",                                                  "dog_bark"),
    (r"insect|cricket|\bbugs?\b|beetle",                               "insect"),
    (r"bird|crow|caw|rooster|\bcock\b|\bhen\b|cluck|duck|cuckoo|sparrow|squawk|twitter|chirp", "bird"),
    (r"\bcow\b|goat|\bcat\b|meow|buffalo|frog|squirrel|animal",        "animal"),
    (r"whistl",                                                        "whistle"),
    (r"music|song|sing|flute|drum|band sound|prayer|mantra|chant|record playing", "music"),
    (r"\bfan\b",                                                       "fan"),           # before "ring": fan whiRRING
    (r"vibrat",                                                        "phone_vibrate"),
    (r"bell|chim",                                                     "bell"),          # before "ring": bell ringing
    (r"\bring(ing|tone|s)?\b|ringtone",                                "phone_ring"),
    (r"beep|notification|message tone|phone sound|mobile sound|phone click", "beep"),
    (r"buzz",                                                          "buzzer"),
    (r"\btv\b|television|speaker|\bmic\b|\bmike\b",                    "tv_speaker"),
    (r"vehicle|traffic|engine|bike|motor|\bcar\b|train|tractor|aeroplane|rickshaw", "vehicle"),
    (r"machine|generator|factory|typing|tick|clock|pump",              "machine"),
]
_RULES = [(re.compile(p), c) for p, c in VAANI_RULES]
DROP = "__drop__"
MIN_EVENTS = 80

def canon_vaani(tag):
    t = re.sub(r"\s+", " ", re.sub(r"[<>\[\]_]", " ", str(tag).lower())).strip()
    for rx, c in _RULES:
        if rx.search(t):
            return c
    return DROP

# NonverbalTTS writes each sound as an emoji inside the transcript (the variation selector U+FE0F is ignored).
TTS_EMOJI = {"🌬": "breath", "🤣": "laugh", "👃": "sniff", "😷": "cough", "🗣": "throat_clear",
             "😤": "sigh", "😖": "groan", "🤧": "sneeze", "😴": "snore", "🐖": "groan"}   # 🐖 = grunt

def parse_tts(text):
    # -> (words, [(class, k)]): normalised words for the aligner, and each sound with the number of words before it
    words, nvs, buf = [], [], ""
    for ch in str(text) + " ":
        if ch in TTS_EMOJI or ch.isspace():
            words += [w.strip("'") for w in re.findall(r"[a-z']+", buf.lower()) if w.strip("'")]
            buf = ""
            if ch in TTS_EMOJI: nvs.append((TTS_EMOJI[ch], len(words)))
        else:
            buf += ch
    return words, nvs

SLR_MAP = {"coughing": "cough", "yawning": "yawn", "throat-clearing": "throat_clear", "sighing": "sigh",
           "lip-smacking": "lip_smack", "lip-popping": "lip_smack", "panting": "breath", "crying": "cry",
           "laughing": "laugh", "sneezing": "sneeze", "nose-blowing": "nose_blow", "moaning": "groan",
           "screaming": "scream", "tongue-clicking": "click", "teeth-chattering": None, "teeth-grinding": None}

# AudioSet ontology names -> canonical tags (first match wins). Unmatched events (speech, impacts, ...) are ignored.
AS_RULES = [
    (r"speech synthesizer|howl \(wind\)|sliding door|sanding|bellow|crowd|growling", None),   # false friends
    (r"^(baby cry|infant cry)",                                         "baby_cry"),
    (r"baby laughter|^laughter|giggle|snicker|belly laugh|chuckle",     "laugh"),
    (r"^crying|sobbing|^whimper$",                                      "cry"),
    (r"^cough",                                                         "cough"),
    (r"^sneeze",                                                        "sneeze"),
    (r"^sniff",                                                         "sniff"),
    (r"^snoring",                                                       "snore"),
    (r"^sigh",                                                          "sigh"),
    (r"^gasp",                                                          "gasp"),
    (r"throat clearing",                                                "throat_clear"),
    (r"^yawn",                                                          "yawn"),
    (r"^(groan|grunt)",                                                 "groan"),
    (r"^(screaming|shriek|yell)",                                       "scream"),
    (r"^(breathing|pant|wheeze)",                                       "breath"),
    (r"^whispering",                                                    "whisper"),
    (r"^shout|battle cry",                                              "shout"),
    (r"^chewing",                                                       "chew"),
    (r"^burping",                                                       "burp"),
    (r"^hiccup",                                                        "hiccup"),
    (r"nose blowing",                                                   "nose_blow"),
    (r"lip smack",                                                      "lip_smack"),
    (r"^child|^children",                                               "child_voice"),
    (r"siren|emergency vehicle|alarm",                                  "siren"),
    (r"horn|honking|^toot$",                                             "horn"),
    (r"^(bark|dog|yip|bow-wow|howl|bay)",                               "dog_bark"),
    (r"insect|cricket|mosquito|^fly|bee, wasp|buzzing insect",          "insect"),
    (r"bird|chirp|tweet|^crow($|ing)|^caw|squawk|pigeon|^coo$|rooster|chicken|duck|quack|^owl", "bird"),
    (r"cattle|moo|goat|bleat|^cat$|meow|purr|frog|croak|horse|neigh|^pig|oink|sheep|animal", "animal"),
    (r"whistl",                                                         "whistle"),
    (r"television|^radio",                                              "tv_speaker"),
    (r"mechanical fan|air conditioning",                                "fan"),
    (r"vibrating alert",                                                "phone_vibrate"),
    (r"telephone bell|ringtone|^telephone$",                            "phone_ring"),
    (r"bell\b|^ding|wind chime",                                                 "bell"),
    (r"beep|bleep|dtmf",                                                "beep"),
    (r"^buzz",                                                          "buzzer"),
    (r"music|singing|song|choir|orchestra|guitar|piano|drum|flute|violin|synthesizer|keyboard \(musical\)|instrument", "music"),
    (r"vacuum cleaner|washing machine|dishwasher|blender|hair dryer|electric shaver|sewing machine|microwave|printer|typing|computer keyboard|clock|tick-tock", "machine"),
    (r"^(clapping|applause)",                                           "applause"),
    (r"^(cheering|whoop)",                                              "cheer"),
    (r"^crowd$|hubbub|speech babble",                                   "crowd"),
    (r"footsteps|^run$",                                                "footsteps"),
    (r"^wind$|^wind noise",                                                 "wind"),
    (r"thunder",                                                        "thunder"),
    (r"^rain|^patter",                                                  "rain"),
    (r"^water|splash|^pour|^drip|trickle|^stream$|^liquid|gurgling|toilet flush|waves|^ocean", "water"),
    (r"^knock",                                                         "knock"),
    (r"^door$|^slam",                                                   "door"),
    (r"^dishes|cutlery|chink, clink",                                   "dishes"),
    (r"^shatter|^breaking",                                     "glass_break"),
    (r"gunshot|machine gun|cap gun|artillery",                          "gunshot"),
    (r"explosion|firecracker|fireworks",                                "explosion"),
    (r"^hammer|power tool|^drill|chainsaw|^sawing|^tools$",             "construction"),
    (r"vehicle|^car|truck|^bus$|helicopter|motorboat|subway|air brake|motorcycle|traffic|engine|idling|revving|race car|train|railroad|rail transport|aircraft|tractor", "vehicle"),
]
_AS_RULES = [(re.compile(p), c) for p, c in AS_RULES]
def canon_as(name):
    t = str(name).lower()
    for rx, c in _AS_RULES:
        if rx.search(t): return c
    return None

# Noise datasets: labelled ambient sounds, or NOISE = unlabelled background for the augmentation pool.
NOISE = "__noise__"
DNC_MAP = {"TV": "tv_speaker", "Radio": "tv_speaker", "Music": "music", "autos": "vehicle",
           "WashingMachine": "machine", "WaterHeater": "machine", "Dishwasher": "machine", "CoffeeMachine": "machine",
           "VacuumCleaner": "machine", "Microwave": "machine", "quiet": NOISE, "people": NOISE}
DEMAND_MAP = {"STRAFFIC": "vehicle", "TBUS": "vehicle", "TCAR": "vehicle", "TMETRO": "vehicle", "DWASHING": "machine"}
                                                   # every other DEMAND environment -> NOISE

# The event list. Speaker sounds go inline as [tag]; background sounds wrap words as <tag> ... </tag>.
# A class is trained only if it is listed here AND has >= MIN_EVENTS events across the datasets (Step 2).
SPEAKER_EVENTS = ["breath", "sigh", "sniff", "laugh", "cough", "throat_clear", "cry", "gasp", "yawn", "snore",
                  "sneeze", "nose_blow", "lip_smack", "groan", "scream", "whisper", "shout", "chew", "burp", "hiccup",
                  "click"]                     # click: OpenSLR 99 only (47 clips) -> below MIN_EVENTS for now
BACKGROUND_EVENTS = ["horn", "vehicle", "siren", "bird", "insect", "dog_bark", "animal", "child_voice", "baby_cry",
                     "music", "tv_speaker", "machine", "fan", "phone_ring", "phone_vibrate", "beep", "bell", "whistle",
                     "buzzer", "applause", "cheer", "crowd", "footsteps", "wind", "rain", "thunder", "water", "knock",
                     "door", "dishes", "glass_break", "gunshot", "explosion", "construction"]
EVENTS = SPEAKER_EVENTS + BACKGROUND_EVENTS
INLINE = set(SPEAKER_EVENTS)
''')

# ----------------------------------------------------------------------------------------------- model def
md(r'''
## Step 1b: Model definition (log-mel → CNN → Transformer → per-frame sigmoid, one token per 40 ms)
It's defined early because Step 3d already uses the current model to clean the speech beds. The architecture is the same as `sed/model.py`, so new checkpoints work with `scripts/infer.py` unchanged.
''')

code(r'''
class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=512, win_length=400, hop_length=160,
            n_mels=N_MELS, f_min=50, f_max=8000, power=2.0)
    def forward(self, wav):                      # (B, T) -> (B, M, T'), always fp32
        x = 10 * torch.log10(self.m(wav.float()) + 1e-9)
        return ((x + 60) / 20).clamp(-1, 3)

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())
    def forward(self, x): return self.net(x)

class SED(nn.Module):
    def __init__(self, n_classes, n_mels=N_MELS, base=48, d=256, heads=8, layers=6):
        super().__init__()
        self.front = nn.Sequential(                    # stride = (freq, time); input (B, 1, M, T) at 10 ms
            ConvBlock(1, base),
            ConvBlock(base, base,         stride=(2, 1)),   # freq /2
            ConvBlock(base, 2 * base,     stride=(2, 1)),   # freq /4
            ConvBlock(2 * base, 2 * base, stride=(2, 2)),   # freq /8,  time /2 -> 20 ms
            ConvBlock(2 * base, 4 * base, stride=(2, 2)))   # freq /16, time /4 -> 40 ms
        self.proj = nn.Linear(4 * base * (n_mels // 16), d)
        pe = torch.zeros(4000, d)
        pos = torch.arange(4000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(1e4) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)
    def forward(self, mel):                    # (B, M, T')
        f = self.front(mel.unsqueeze(1))       # (B, C, M', T'/4)
        B, Ch, M, T = f.shape
        h = self.proj(f.permute(0, 3, 1, 2).reshape(B, T, Ch * M)) + self.pe[:T]
        return self.head(self.enc(h))          # (B, T'/4, n_classes)

feats = LogMel().to(dev)
T_SAMP = int(DUR * SR)
N_FR = feats(torch.zeros(1, T_SAMP, device=dev)).shape[-1]        # 801 mel frames per window

def load_old_model():
    # The current model (checkpoints/best.pt) with its balanced thresholds, or None if it isn't available.
    if not (OLD_DIR and os.path.exists(f"{OLD_DIR}/best.pt")): return None
    ck = torch.load(f"{OLD_DIR}/best.pt", map_location=dev, weights_only=True)
    m = SED(len(ck["classes"])).to(dev).eval(); m.load_state_dict(ck["model"])
    thr_path = f"{OLD_DIR}/thresholds.json"
    thr = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    return m, ck["classes"], np.array([min(thr.get(c, 0.9), 0.99) for c in ck["classes"]])
print(f"{N_FR} mel frames per {DUR:.0f}s window | current model available: {os.path.exists(f'{OLD_DIR}/best.pt')}")
''')

# ----------------------------------------------------------------------------------------------- index
md(r'''
## Step 2: Metadata index (no audio) and class list
This step reads only the label and timestamp columns and records which parquet row group holds each clip. It's cached in `CACHE/index/`, and smoke and full runs share it. NV38k validation uses a fixed row hash, so a model trained earlier with this pipeline is scored on clips it never saw. Vaani validation is held out by district.
''')

code(r'''
IDX = os.path.join(CACHE, "index"); os.makedirs(IDX, exist_ok=True)

def build_index(repo, cols, out, threads=16):
    if os.path.exists(out):
        return pd.read_parquet(out)
    def one(rel):
        with pq_open(repo, rel, 1 << 20) as fh:
            pf = pq.ParquetFile(fh); md_ = pf.metadata
            d = pf.read(columns=cols).to_pandas()
        rg = np.concatenate([np.full(md_.row_group(i).num_rows, i) for i in range(md_.num_row_groups)])
        d["file"] = rel; d["rg"] = rg; d["row"] = d.groupby("rg").cumcount()
        d["rg_mb"] = np.array([md_.row_group(i).total_byte_size / 1e6 for i in range(md_.num_row_groups)])[rg]
        return d
    files = pq_files(repo)
    with ThreadPoolExecutor(threads) as ex:
        df = pd.concat(list(tqdm(ex.map(one, files), total=len(files), desc=repo)), ignore_index=True)
    if "NoiseSubCategoryTimeStamp" in df:    # nested list -> JSON string (portable, tiny)
        df["ts"] = [json.dumps([{k: e[k] for k in ("tag", "start", "end")} for e in (x if x is not None else [])])
                    for x in df.pop("NoiseSubCategoryTimeStamp")]
    if "events" in df:                       # AudioSet-Strong: [{event_name, start, end}] -> JSON string
        df["ts"] = [json.dumps([[e["event_name"], float(e["start"]), float(e["end"])] for e in (x if x is not None else [])])
                    for x in df.pop("events")]
    if "non_verbal_region" in df:
        r = np.stack(df.pop("non_verbal_region").values); df["ev_s"], df["ev_e"] = r[:, 0], r[:, 1]
    df.to_parquet(out)
    return df

def is_val(key, pct=10):  # stable hash split
    return zlib.crc32(str(key).encode()) % 100 < pct

nv_idx = build_index(NV_REPO, ["duration", "non_verbal_region", "label", "language", "source"], f"{IDX}/nv.parquet")
tts_idx = build_index(TTS_REPO, ["index", "Result", "duration", "data_name", "dnsmos", "speaker_id"], f"{IDX}/nvtts.parquet")
va_idx = (build_index(VA_REPO, ["duration", "language", "district", "annotationQuality", "NoiseSubCategoryTimeStamp"],
                      f"{IDX}/vaani.parquet") if HAVE_VAANI else
          pd.DataFrame(columns=["file", "rg", "row", "duration", "district", "annotationQuality", "ts"]))
as_idx = build_index(AS_REPO, ["video_id", "events"], f"{IDX}/audioset.parquet")
if not HAVE_VAANI: print("WARNING: Vaani is not available: background-noise classes will be missing. Set HF_TOKEN.")

backup(IDX)
slr_root = f"{DATA}/openslr99/NonverbalVocalization"
slr_meta = json.load(open(f"{slr_root}/Nonverbal_Vocalization.json"))
slr_idx = pd.DataFrame([dict(path=f"{slr_root}/{lab}/{fn}", label=lab, speaker=m["speakerID"])
                        for lab, files in slr_meta.items() for fn, m in files.items()
                        if os.path.exists(f"{slr_root}/{lab}/{fn}")])

# labels -> canonical classes
nv_idx["cls"] = nv_idx.label.map(NV_MAP)
va_idx["events"] = [[(canon_vaani(e["tag"]), float(e["start"]), float(e["end"])) for e in json.loads(s)]
                    for s in va_idx.ts]
tts_idx["nvs"] = [parse_tts(t)[1] for t in tts_idx.Result]
as_idx["events"] = [[(canon_as(n), s, e) for n, s, e in json.loads(t)] for t in as_idx.ts]
slr_idx["cls"] = slr_idx.label.map(SLR_MAP)

# splits (val is held out from training everywhere)
nv_idx["split"] = ["val" if is_val(f"{os.path.basename(f)}/{g}/{r}") else "train"     # same hash as the current model
                   for f, g, r in zip(nv_idx.file, nv_idx.rg, nv_idx.row)]
va_val_district = va_idx.district.map(is_val).astype(bool)
va_idx["split"] = np.where(~va_val_district, "train",
                           np.where(va_idx.annotationQuality == "verified_timestamps", "val", "skip"))
va_idx.loc[va_idx.annotationQuality == "no_timestamps", "split"] = "skip"
as_idx["split"] = ["val" if os.path.basename(f).startswith("test") else "train" for f in as_idx.file]   # official eval set
tts_idx["split"] = ["val" if f.split("/")[1] in ("dev", "test") else "train" for f in tts_idx.file]   # official splits
slr_idx["split"] = ["val" if is_val(s, 15) else "train" for s in slr_idx.speaker]

with zipfile.ZipFile(f"{DATA}/dnc/DNC.zip") as z:
    dnc_idx = pd.DataFrame({"path": sorted(n for n in z.namelist() if n.lower().endswith(".wav") and "MACOSX" not in n and "/._" not in n)})
dnc_idx["kind"] = [re.split(r" _ |_\d", os.path.basename(p))[0].strip() for p in dnc_idx.path]
dnc_idx["cls"] = dnc_idx.kind.map(DNC_MAP)
dnc_idx["split"] = ["val" if is_val(re.sub(r"_\d+\.wav$", "", os.path.basename(p))) else "train" for p in dnc_idx.path]  # by session
demand_envs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{DATA}/demand/*/ch01.wav"))

n_by = {
    "nonverbal38k": collections.Counter(nv_idx.cls.dropna()),
    "vaani": collections.Counter(c for evs in va_idx.events for c, _, _ in evs if c not in (None, DROP)),
    "nonverbaltts": collections.Counter(c for nvs in tts_idx.nvs for c, _ in nvs),
    "audioset": collections.Counter(c for evs in as_idx.events for c, _, _ in evs if c),
    "openslr99": collections.Counter(slr_idx.cls.dropna()),
    "dnc": collections.Counter(c for c in dnc_idx.cls.dropna() if c != NOISE),
    "demand": collections.Counter(DEMAND_MAP[e] for e in demand_envs if e in DEMAND_MAP),   # environments
}
counts = sum(n_by.values(), collections.Counter())
unknown = sorted(set(counts) - set(EVENTS))
assert not unknown, f"mapped to tags missing from EVENTS: {unknown}"
classes = sorted(c for c in EVENTS if counts.get(c, 0) >= MIN_EVENTS)
cls2idx = {c: i for i, c in enumerate(classes)}
C = len(classes)
# which classes each real-window source labels (a class is only scored on sources that label it)
LABELS = {"nv": set(n_by["nonverbal38k"]) & set(classes), "vaani": set(n_by["vaani"]) & set(classes),
          "nvtts": set(n_by["nonverbaltts"]) & set(classes), "audioset": set(n_by["audioset"]) & set(classes)}

print(f"{C} of {len(EVENTS)} listed events trained | dropped (<{MIN_EVENTS} events): "
      f"{ {c: counts.get(c, 0) for c in EVENTS if c not in cls2idx} }")
tab = pd.DataFrame({k: {c: v.get(c, 0) for c in classes} for k, v in n_by.items()})
tab["inline"] = [c in INLINE for c in classes]
print(tab.sort_values("vaani", ascending=False).to_string())
print("clips:", {"nv": len(nv_idx), "vaani": len(va_idx), "nvtts": len(tts_idx), "slr99": len(slr_idx),
                 "audioset": len(as_idx), "dnc": len(dnc_idx), "demand environments": len(demand_envs)}, "|", mem())
''')

# ----------------------------------------------------------------------------------------------- banks
md(r'''
## Step 3: Audio banks from NV38k and Vaani
A greedy pass picks the row groups that best fill each class's quota, with rare classes weighted up. Locally the budget is unlimited, so it reads whatever fills the quotas. The chosen row groups are decoded in parallel, resampled to 16 kHz and appended as int16 to memory-mapped stores:

| store | contents |
|---|---|
| `ev_main` | event crops: NV `non_verbal_region` and Vaani `start/end`, with a ±30 ms margin |
| `beds_main` | speech-only segments: NV audio away from the labelled event |
| `real_main` | whole clips with all their timestamps: Vaani (all events mapped) and **NV38k (new)** |
''')

code(r'''
def select_rowgroups(df, row_classes, cap, budget, n_val_rg=0):
    # Greedy: repeatedly take the row group that fills the most remaining per-class quota,
    # then add the n_val_rg row groups richest in held-out validation clips.
    if not len(df): return []
    keys = list(df.groupby(["file", "rg"]).groups)
    pos = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(keys), C), np.float32)
    for k, cs in zip(zip(df.file, df.rg), row_classes):
        for c in cs:
            if c in cls2idx: M[pos[k], cls2idx[c]] += 1
    w = 1.0 / np.minimum(cap, M.sum(0)).clip(min=1)      # rare classes count as much as common ones
    need, picked = np.full(C, float(cap)), []
    for _ in range(min(budget, len(keys))):
        gain = (np.minimum(M, need) * w).sum(1)
        i = int(gain.argmax())
        if gain[i] <= 0: break
        picked.append(keys[i]); need = np.maximum(need - M[i], 0); M[i] = 0
    n_val = df[df.split == "val"].groupby(["file", "rg"]).size().drop(picked, errors="ignore")
    return picked + list(n_val.sort_values(ascending=False).index[:n_val_rg])

nv_rows = [[c] if s == "train" else [] for c, s in zip(nv_idx.cls, nv_idx.split)]
va_rows = [[c for c, _, _ in e] if s != "skip" else [] for e, s in zip(va_idx.events, va_idx.split)]
# NV: also add row groups rich in val clips, so NV has held-out real windows (2 per 10 train row groups)
nv_pick = select_rowgroups(nv_idx, nv_rows, CFG["bank_cap"], CFG["nv_rg_budget"],
                           max(1, min(CFG["nv_rg_budget"], 10**4) // 5))
va_pick = select_rowgroups(va_idx, va_rows, CFG["bank_cap"], CFG["va_rg_budget"], CFG["va_val_rg"])
print(f"NV: {len(nv_pick)} row groups   Vaani: {len(va_pick)} row groups")
''')

code(r'''
class AudioStore:
    # Append-only int16 PCM file plus a JSON sidecar; nothing accumulates in RAM.
    def __init__(self, path):
        self.path, self.f, self.meta, self.n = path, open(path + ".part", "wb"), [], 0
    def add(self, x, **m):
        x16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        self.f.write(x16.tobytes()); self.meta.append(dict(off=self.n, len=len(x16), **m)); self.n += len(x16)
    def close(self):
        self.f.close(); os.replace(self.path + ".part", self.path + ".pcm")
        json.dump(self.meta, open(self.path + ".json", "w"))

class Clips:
    # Read side of one or more AudioStores: memory-mapped, so DataLoader workers share pages and never copy.
    def __init__(self, *paths):
        self.meta, self.pcm = [], []
        for p in paths:
            if not os.path.exists(p + ".json"): continue
            self.pcm.append(np.memmap(p + ".pcm", np.int16, "r") if os.path.getsize(p + ".pcm") else np.zeros(0, np.int16))
            self.meta += [dict(m, store=len(self.pcm) - 1) for m in json.load(open(p + ".json"))]
    def __len__(self): return len(self.meta)
    def get(self, i):
        m = self.meta[i]; return self.pcm[m["store"]][m["off"]: m["off"] + m["len"]].astype(np.float32) / 32767

def stream_rowgroups(repo, picks, workers=4):
    # Yields (file, rg, [wav bytes]) for each picked row group; at most 2*workers row groups in memory.
    def fetch(k):
        with pq_open(repo, k[0]) as fh:
            col = pq.ParquetFile(fh).read_row_group(int(k[1]), columns=["audio"]).column("audio")
        return k, [a["bytes"] for a in col.to_pylist()]
    it = iter(picks)
    with ThreadPoolExecutor(workers) as ex:
        q = collections.deque(ex.submit(fetch, k) for k in itertools.islice(it, 2 * workers))
        while q:
            k, blobs = q.popleft().result()
            nxt = next(it, None)
            if nxt is not None: q.append(ex.submit(fetch, nxt))
            yield k, blobs

N_BAD = collections.Counter()

def decode(b, where=""):
    # Returns None for clips libsndfile can't read (Vaani contains e.g. an HTML page saved as "audio").
    try:
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    except Exception as e:
        N_BAD[where] += 1
        if sum(N_BAD.values()) <= 5: print(f"skipping undecodable clip {where}: {str(e)[:80]}")
        return None
    if len(x) == 0 or not np.isfinite(x).all():
        N_BAD[where] += 1; return None
    x = torch.from_numpy(x.mean(1))
    if sr != SR: x = torchaudio.functional.resample(x, sr, SR)
    return x.numpy()

DEC = ThreadPoolExecutor(CFG["decode_threads"])
def decode_rows(blobs, rows, where):        # decode only the rows we need, in parallel
    return dict(zip(rows, DEC.map(lambda r: decode(blobs[r], f"{where}/row{r}"), rows)))

BANK = os.path.join(RUN, "audio"); os.makedirs(BANK, exist_ok=True)

def build_main_banks():
    if all(os.path.exists(f"{BANK}/{n}.pcm") for n in ("ev_main", "beds_main", "real_main")):
        return
    ev, beds_, real_ = (AudioStore(f"{BANK}/{n}") for n in ("ev_main", "beds_main", "real_main"))
    n_ev, n_bed, n_real = collections.Counter(), collections.Counter(), collections.Counter()
    caps = {"train": CFG["bank_cap"], "val": max(2, CFG["bank_cap"] // 8)}
    real_caps = {("nv", "train"): CFG["real_cap_nv"], ("nv", "val"): CFG["real_cap_nv"] // 8,
                 ("vaani", "train"): CFG["real_cap_vaani"], ("vaani", "val"): CFG["real_cap_vaani"] // 8}
    m = 0.03                                                      # crop margin (s)

    nv_by = nv_idx.set_index(["file", "rg", "row"]).sort_index()
    for (f, g), blobs in tqdm(stream_rowgroups(NV_REPO, nv_pick), total=len(nv_pick), desc="NV38k"):
        plan = {}
        for r in range(len(blobs)):
            row = nv_by.loc[(f, g, r)]
            c, sp = row.cls, row.split
            take_ev = c in cls2idx and n_ev[("nv", c, sp)] < caps[sp] and 0.15 <= row.ev_e - row.ev_s <= 5.0
            take_bed = n_bed[sp] < CFG["bed_cap"] // (1 if sp == "train" else 8)
            take_real = c in cls2idx and row.duration <= 20 and n_real[("nv", sp)] < real_caps[("nv", sp)]
            if take_ev or take_bed or take_real:
                plan[r] = (row, take_ev, take_bed, take_real)
                n_ev[("nv", c, sp)] += take_ev; n_bed[sp] += take_bed; n_real[("nv", sp)] += take_real
        for r, x in decode_rows(blobs, list(plan), f"{f}/rg{g}").items():
            if x is None: continue
            row, take_ev, take_bed, take_real = plan[r]
            c, sp, s, e = row.cls, row.split, row.ev_s, row.ev_e
            if take_ev:
                ev.add(x[int(max(0, s - m) * SR): int((e + m) * SR)], label=c, split=sp, src="nv")
            if take_bed:            # speech bed: longest stretch away from the event (other events unlabeled)
                a, z = x[: int(max(0, s - 0.3) * SR)], x[int((e + 0.3) * SR):]
                seg = a if len(a) >= len(z) else z
                if len(seg) >= 2 * SR: beds_.add(seg[: int(10 * SR)], split=sp, src="nv")
            if take_real:
                real_.add(x, events=[[c, float(s), float(e)]], split=sp, src="nv")

    va_by = va_idx.set_index(["file", "rg", "row"]).sort_index()
    for (f, g), blobs in tqdm(stream_rowgroups(VA_REPO, va_pick), total=len(va_pick), desc="Vaani"):
        plan = {}
        for r in range(len(blobs)):
            row = va_by.loc[(f, g, r)]
            sp = row.split
            if sp == "skip": continue
            evs = [(c, s, e) for c, s, e in row.events if c is not None]
            clean = all(c in cls2idx for c, _, _ in evs) and len(evs) > 0
            want_crop = [(c, s, e) for c, s, e in evs if c in cls2idx and n_ev[("vaani", c, sp)] < caps[sp]
                         and 0.15 <= e - s <= (3.0 if c in INLINE else 8.0)]
            want_real = clean and row.duration <= 20 and n_real[("vaani", sp)] < real_caps[("vaani", sp)]
            if want_crop or want_real:
                plan[r] = (row, evs, want_crop, want_real)
                for c, _, _ in want_crop: n_ev[("vaani", c, sp)] += 1
                n_real[("vaani", sp)] += want_real
        for r, x in decode_rows(blobs, list(plan), f"{f}/rg{g}").items():
            if x is None: continue
            row, evs, want_crop, want_real = plan[r]
            for c, s, e in want_crop:
                ev.add(x[int(max(0, s - m) * SR): int((e + m) * SR)], label=c, split=row.split, src="vaani")
            if want_real:
                real_.add(x, events=[[c, s, e] for c, s, e in evs], split=row.split, src="vaani", district=row.district)
        gc.collect()
    for st in (ev, beds_, real_): st.close()
    print("speech beds", dict(n_bed), "| real clips", dict(n_real), "| undecodable clips skipped:", sum(N_BAD.values()))

build_main_banks()
print(mem())
''')

# ----------------------------------------------------------------------------------------------- nvtts
md(r'''
## Step 3a: AudioSet-Strong (streamed): real annotated windows and event crops
AudioSet clips are 10 s YouTube excerpts in which annotators marked every sound with a start and an end time. Its event names are mapped to our tags by `AS_RULES`; everything else (speech, impacts, …) is simply not one of our classes. The 121 GB dataset is **streamed** from the Hub, even in the local copy: a greedy pass picks the row groups (72 clips, ~89 MB each) that best cover our classes, and only the chosen clips are stored, as whole real windows (`real_as`) and crops (`ev_as`). That is about 0.3 MB per clip, so roughly 9 GB for 30k clips. Validation uses the official AudioSet evaluation set.
''')

code(r'''
as_rows = [[c for c, _, _ in e] if sp == "train" else [] for e, sp in zip(as_idx.events, as_idx.split)]
as_pick = select_rowgroups(as_idx, as_rows, CFG["bank_cap"], CFG["as_rg_budget"], CFG["as_val_rg"])
print(f"AudioSet: {len(as_pick)} row groups, ~{as_idx.drop_duplicates(['file', 'rg']).set_index(['file', 'rg']).rg_mb.loc[as_pick].sum()/1e3:.0f} GB to stream")

def build_as_banks():
    if all(os.path.exists(f"{BANK}/{n}.pcm") for n in ("ev_as", "real_as")): return
    ev, real_ = AudioStore(f"{BANK}/ev_as"), AudioStore(f"{BANK}/real_as")
    n_ev, n_real, n_empty = collections.Counter(), collections.Counter(), collections.Counter()
    caps = {"train": CFG["bank_cap"], "val": max(2, CFG["bank_cap"] // 8)}
    rcap = {"train": CFG["real_cap_as"], "val": CFG["real_cap_as"] // 8}
    by = as_idx.set_index(["file", "rg", "row"]).sort_index()
    for (f, g), blobs in tqdm(stream_rowgroups(AS_REPO, as_pick), total=len(as_pick), desc="AudioSet"):
        plan = {}
        for r in range(len(blobs)):
            row = by.loc[(f, g, r)]; sp = row.split
            evs = [(c, s, e) for c, s, e in row.events if c in cls2idx]
            crops = [(c, s, e) for c, s, e in evs if n_ev[(c, sp)] < caps[sp] and 0.15 <= e - s <= (3.0 if c in INLINE else 8.0)]
            take_real = bool(n_real[sp] < rcap[sp] and (evs or n_empty[sp] < rcap[sp] // 10))   # ~10% event-free windows
            if crops or take_real:
                plan[r] = (row, evs, crops, take_real)
                for c, _, _ in crops: n_ev[(c, sp)] += 1
                n_real[sp] += take_real; n_empty[sp] += take_real and not evs
        for r, x in decode_rows(blobs, list(plan), f"{f}/rg{g}").items():
            if x is None: continue
            row, evs, crops, take_real = plan[r]
            for c, s, e in crops:
                ev.add(x[int(max(0, s - 0.03) * SR): int((e + 0.03) * SR)], label=c, split=row.split, src="audioset")
            if take_real:
                real_.add(x, events=[[c, s, e] for c, s, e in evs], split=row.split, src="audioset")
        gc.collect()
    ev.close(); real_.close()
    print("AudioSet real clips", dict(n_real), "(event-free:", dict(n_empty), ") | crops", sum(n_ev.values()))

build_as_banks()
print(mem())
''')

md(r'''
## Step 3b: NonverbalTTS: forced alignment → timestamps
NonverbalTTS has no timestamps. Each emoji sits between two words of the transcript, for example `…amazing. 🌬️ The fact…`. Its dataset paper says the emoji were placed using a detector's timestamps and a forced aligner. So:
1. **Align** the words to the audio with the MMS_FA model from torchaudio and a small CTC Viterbi aligner written in plain PyTorch, which matches torchaudio's own aligner exactly. `*` wildcards at both ends absorb speech the transcript left out.
2. **Locate** each sound in the pause between the word before its emoji and the word after. Inside that pause, the event is the most energetic stretch above the clip's noise floor (+4 dB). If nothing rises above the floor and the pause is short enough, the whole pause is used. Sounds whose pause is shorter than ~0.2 s (e.g. laughing *while* talking) aren't located.
3. Clips where **every** sound was located become real annotated windows. Located sounds also become event crops. Clips without any sound are English speech beds and negative windows.

In the Experiment 1 run, 6,360 of the 8,620 annotated sounds (74%) were located and 4,549 clips became real windows. Expect roughly 1 h for all 6.3k clips on a T4 (about 0.5 s per clip); it runs only once.
''')

code(r'''
MAXLEN = dict(breath=1.5, laugh=4.0, sniff=1.0, cough=2.0, throat_clear=1.5, sigh=2.0, groan=2.5,
              sneeze=2.0, snore=3.0)                    # longest plausible duration per class (s)

def frame_db(x, hop=320):                              # 20 ms frame energy in dB
    n = len(x) // hop
    return 10 * np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)

def locate(c, g0, g1, db, floor, shrink=0.06):
    # The event inside the pause [g0, g1] (s): the most energetic stretch above floor + 4 dB; the whole pause
    # if nothing rises above the floor and the pause is short; None if the pause is too short.
    g0, g1 = g0 + shrink, g1 - shrink                  # word edges bleed into the pause
    if g1 - g0 < 0.10: return None
    i0, i1 = int(g0 / 0.02), int(np.ceil(g1 / 0.02)); d = db[i0:i1]
    idx = np.flatnonzero(d > floor + 4)
    if not len(idx):
        return (round(g0, 3), round(g1, 3)) if 0.15 <= g1 - g0 <= MAXLEN.get(c, 2.0) else None
    runs = [[idx[0], idx[0] + 1]]
    for i in idx[1:]:
        if i - runs[-1][1] <= 5: runs[-1][1] = i + 1    # join runs less than 100 ms apart
        else: runs.append([i, i + 1])
    a, b = runs[int(np.argmax([(10 ** (d[a:b] / 10)).sum() for a, b in runs]))]
    s, e = max(g0, (i0 + a) * 0.02 - 0.04), min(g1, (i0 + b) * 0.02 + 0.04)
    if e - s > MAXLEN.get(c, 2.0):                     # too long: keep MAXLEN around the loudest frame
        pk = (i0 + a + int(np.argmax(d[a:b]))) * 0.02
        s = max(s, pk - MAXLEN.get(c, 2.0) / 2); e = min(e, s + MAXLEN.get(c, 2.0))
    return (round(float(s), 3), round(float(e), 3)) if e - s >= 0.1 else None

def ctc_word_spans(em, toks, blank=0):
    # CTC forced alignment (Viterbi) in plain PyTorch, so it doesn't depend on torchaudio's compiled aligner,
    # which newer torchaudio releases are phasing out. Gives the same word spans as torchaudio's aligner.
    # em: (T, V) log-probs; toks: [[token ids] per word] -> [(start_frame, end_frame, mean prob)] per word
    flat = [t for w in toks for t in w]
    ext = [blank]
    for t in flat: ext += [t, blank]
    S, T, NEG = len(ext), em.shape[0], -1e30
    if T < len(flat): raise RuntimeError("audio too short for the transcript")
    ext_t = torch.tensor(ext)
    skip = torch.tensor([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
    pad1, pad2 = torch.full((1,), NEG), torch.full((2,), NEG)
    alpha = torch.full((S,), NEG); alpha[0] = em[0, blank]; alpha[1] = em[0, ext[1]]
    bp = torch.zeros((T, S), dtype=torch.long)       # 0: stay, 1: from s-1, 2: skip a blank
    for t in range(1, T):
        cand = torch.stack([alpha, torch.cat([pad1, alpha[:-1]]), torch.where(skip, torch.cat([pad2, alpha[:-2]]), NEG)])
        best, bp[t] = cand.max(0)
        alpha = best + em[t, ext_t]
    s = S - 1 if alpha[S - 1] > alpha[S - 2] else S - 2
    path = torch.empty(T, dtype=torch.long)
    for t in range(T - 1, -1, -1):
        path[t] = s; s -= int(bp[t, s])
    out, k = [], 0
    for w in toks:
        fr = torch.cat([(path == 2 * (k + i) + 1).nonzero().flatten() for i in range(len(w))]); k += len(w)
        out.append((int(fr.min()), int(fr.max()) + 1, float(em[fr, ext_t[path[fr]]].exp().mean())))
    return out

def build_tts_banks():
    if all(os.path.exists(f"{BANK}/{n}.pcm") for n in ("ev_tts", "beds_tts", "real_tts")):
        return
    bundle = torchaudio.pipelines.MMS_FA
    fa = bundle.get_model(with_star=True).to(dev).eval()
    tok = bundle.get_tokenizer()

    def align(x, words):
        with torch.inference_mode():
            em, _ = fa(torch.from_numpy(x)[None].to(dev))
        spans = ctc_word_spans(em[0].float().cpu(), tok(["*"] + words + ["*"]))[1:-1]
        r = len(x) / em.shape[1] / SR                  # seconds per emission frame (20 ms)
        return [(a * r, b * r) for a, b, _ in spans], float(np.mean([sc for _, _, sc in spans]))

    ev, beds_, real_ = (AudioStore(f"{BANK}/{n}") for n in ("ev_tts", "beds_tts", "real_tts"))
    st, n_done = collections.Counter(), collections.Counter()
    rows = tts_idx[tts_idx.duration <= 30]
    if CFG["tts_cap"]:                                 # smoke: a few row groups per split
        rows = pd.concat([rows[rows.split == sp].groupby(["file", "rg"]).head(10**6).head(CFG["tts_cap"][sp])
                          for sp in ("train", "val")])
    by = {k: g.set_index("row") for k, g in rows.groupby(["file", "rg"])}
    for (f, g), blobs in tqdm(stream_rowgroups(TTS_REPO, list(by)), total=len(by), desc="NonverbalTTS align"):
        rr = by[(f, g)]
        for r, x in decode_rows(blobs, list(rr.index), f"{f}/rg{g}").items():
            if x is None: continue
            row = rr.loc[r]; sp = row.split
            words, nvs = parse_tts(row.Result)
            if not words: st["no words"] += 1; continue
            try:
                spans, score = align(x, words)
            except Exception:
                st["alignment failed"] += 1; continue
            if score < 0.35: st["poor alignment"] += 1; continue
            dur, db = len(x) / SR, frame_db(x)
            floor = float(np.percentile(db, 10))
            events, complete = [], True
            for c, k in nvs:
                g0 = spans[k - 1][1] if k > 0 else max(0.0, spans[0][0] - MAXLEN.get(c, 2.0) - 0.3)
                g1 = spans[k][0] if k < len(spans) else min(dur, spans[-1][1] + MAXLEN.get(c, 2.0) + 0.3)
                se = locate(c, g0, g1, db, floor) if c in cls2idx else None
                st[f"{c}: located" if se else f"{c}: not located"] += 1
                if se: events.append([c, *se])
                else: complete = False
            for c, s, e in events:
                ev.add(x[int(max(0, s - 0.03) * SR): int((e + 0.03) * SR)], label=c, split=sp, src="nvtts")
            if complete:
                real_.add(x, events=events, split=sp, src="nvtts")
                st["real clips"] += 1
            if not nvs and dur >= 2:
                beds_.add(x[: int(10 * SR)], split=sp, src="nvtts"); st["speech beds"] += 1
        gc.collect()
    for s_ in (ev, beds_, real_): s_.close()
    del fa; gc.collect(); torch.cuda.empty_cache()
    print(pd.Series(st).sort_index().to_string())

build_tts_banks()
print(mem())
''')

# ----------------------------------------------------------------------------------------------- slr99
md(r'''
## Step 3c: OpenSLR 99: trim isolated vocalizations into event crops
Each clip is one sound with silence around it, recorded on a phone. The crop runs from the first to the last frame louder than max(noise floor + 8 dB, peak − 40 dB), with 50 ms padding.
''')

code(r'''
def trim_event(x, pad=0.05):
    db = frame_db(x)
    if not len(db): return None
    act = np.flatnonzero(db > max(np.percentile(db, 10) + 8, db.max() - 40))
    if not len(act): return None
    a, b = max(0, int((act[0] * 0.02 - pad) * SR)), int(((act[-1] + 1) * 0.02 + pad) * SR)
    return x[a:b] if b - a >= 0.15 * SR else None

def build_slr_bank():
    if os.path.exists(f"{BANK}/ev_slr.pcm"): return
    st, n = AudioStore(f"{BANK}/ev_slr"), collections.Counter()
    for row in tqdm(slr_idx.itertuples(), total=len(slr_idx), desc="OpenSLR 99"):
        if row.cls not in cls2idx or (CFG["slr_cap"] and n[row.cls] >= CFG["slr_cap"]): continue
        x, sr = sf.read(row.path, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR: x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
        crop = trim_event(x)
        if crop is not None:
            st.add(crop[: int(5 * SR)], label=row.cls, split=row.split, src="slr99"); n[row.cls] += 1
    st.close(); print(dict(n))

build_slr_bank()
''')

md(r'''
## Step 3d: DNC and DEMAND: ambient event crops and a real background-noise pool
- **Labelled recordings** become ambient event crops (`ev_amb`), up to 16 s each. The mixer cuts a random 8 s piece and places it under speech. DNC is split by recording session. DEMAND (5 min per environment) is split by time: the last 20% of each recording is validation. Channels 1 and 9 of the 16-mic array are used.
- **Unlabelled recordings** (DNC quiet/people, and DEMAND kitchen, living room, field, park, river, hallway, meeting, office, cafeteria, restaurant, station, café, square) are cut into 10 s chunks (`noise`). Step 3e screens them, and augmentation adds them under speech as real-world background.
''')

code(r'''
_zips = threading.local()
def read16(path):               # a wav path, or "zip:<member>" inside DNC.zip (one ZipFile handle per thread)
    if path.startswith("zip:"):
        if not hasattr(_zips, "z"): _zips.z = zipfile.ZipFile(f"{DATA}/dnc/DNC.zip")
        path = io.BytesIO(_zips.z.read(path[4:]))
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    return (torchaudio.functional.resample(x, sr, SR) if sr != SR else x).numpy()

def build_noise_banks():
    if all(os.path.exists(f"{BANK}/{n}.pcm") for n in ("ev_amb", "noise")): return
    amb, noise = AudioStore(f"{BANK}/ev_amb"), AudioStore(f"{BANK}/noise")
    n_amb, n_noise = collections.Counter(), collections.Counter()
    caps = {"train": CFG["bank_cap"], "val": max(2, CFG["bank_cap"] // 8)}
    jobs, planned = [], collections.Counter()           # apply the caps before decoding anything
    for r in dnc_idx.itertuples():
        key = "noise" if r.cls == NOISE else (r.cls, r.split)
        cap = (CFG["noise_cap"] or 10**9) if r.cls == NOISE else caps[r.split] if r.cls in cls2idx else 0
        if planned[key] < cap: jobs.append(r); planned[key] += 1
    for r, x in zip(jobs, DEC.map(lambda r: read16("zip:" + r.path), jobs)):
        if len(x) < SR: continue
        if r.cls == NOISE:
            noise.add(x[: 10 * SR], src="dnc", kind=r.kind); n_noise["dnc"] += 1
        else:
            amb.add(x[: 16 * SR], label=r.cls, split=r.split, src="dnc"); n_amb[("dnc", r.cls, r.split)] += 1
    for e in tqdm(demand_envs, desc="DEMAND"):
        for ch in ("ch01", "ch09"):
            x = read16(f"{DATA}/demand/{e}/{ch}.wav")
            cut = int(0.8 * len(x))
            if e in DEMAND_MAP:
                c = DEMAND_MAP[e]
                if c not in cls2idx: continue
                for sp, seg in (("train", x[:cut]), ("val", x[cut:])):
                    for o in range(0, len(seg) - 8 * SR + 1, 8 * SR):
                        if n_amb[("demand", c, sp)] >= caps[sp]: break
                        amb.add(seg[o: o + 8 * SR], label=c, split=sp, src="demand"); n_amb[("demand", c, sp)] += 1
            else:
                for o in range(0, len(x) - 10 * SR + 1, 10 * SR):
                    if CFG["noise_cap"] and n_noise[("demand", e)] >= CFG["noise_cap"] // 4: break
                    noise.add(x[o: o + 10 * SR], src="demand", kind=e); n_noise[("demand", e)] += 1
    amb.close(); noise.close()
    print("ambient crops:", dict(collections.Counter({f"{s}/{c}/{sp}": n for (s, c, sp), n in n_amb.items()})))
    print("noise chunks:", sum(n_noise.values()))

build_noise_banks()
''')

code(r'''
ev_bank = Clips(f"{BANK}/ev_main", f"{BANK}/ev_as", f"{BANK}/ev_tts", f"{BANK}/ev_slr", f"{BANK}/ev_amb")
beds = Clips(f"{BANK}/beds_main", f"{BANK}/beds_tts")
noise_bank = Clips(f"{BANK}/noise")
real = Clips(f"{BANK}/real_main", f"{BANK}/real_as", f"{BANK}/real_tts")
tab = pd.crosstab(pd.Series([m["label"] for m in ev_bank.meta], name="class"),
                  pd.Series([f'{m["src"]}/{m["split"]}' for m in ev_bank.meta], name="crops"))
print(tab.to_string())
missing = [c for c in classes if c not in tab.index]
print(f"classes with no crops: {missing}" if missing else "every class has crops")
print("real clips:", collections.Counter(f'{m["src"]}/{m["split"]}' for m in real.meta))
print("speech beds:", collections.Counter(f'{m["src"]}/{m["split"]}' for m in beds.meta))
for n in os.listdir(BANK):
    if n.endswith(".pcm"): print(f"{n}: {os.path.getsize(f'{BANK}/{n}')/1e9:.2f} GB")
print(mem())
''')

# ----------------------------------------------------------------------------------------------- bed cleaning
md(r'''
## Step 3e: Clean the speech beds and the noise pool with the current model
Speech beds are NV38k audio away from the labelled event, plus NonverbalTTS clips without emoji. They still contain breaths, sniffs and laughs that nobody labelled. Pasting events onto them teaches the model to stay silent on real events, which hurts recall. The background-noise pool can likewise contain horns, birds or laughter. The current model marks frames where a class passes its tuned threshold: vocal classes for the beds, every class for the noise pool. Those frames ±0.3 s are cut, and the longest clean stretch of at least 2 s is kept. Without `checkpoints/best.pt`, everything is kept whole.
''')

code(r'''
@torch.no_grad()
def clean_segments(clips, name, vocal_only):
    out = f"{BANK}/{name}_clean.json"
    if os.path.exists(out): return json.load(open(out))
    old = load_old_model()
    if old is None:
        print(f"current model not found: {name} used as is")
        segs = [[i, 0, m["len"]] for i, m in enumerate(clips.meta)]
        json.dump(segs, open(out, "w")); return segs
    m_old, old_cls, old_thr = old
    use = np.array([c in INLINE or not vocal_only for c in old_cls])
    segs, flagged = [], 0.0
    for i0 in tqdm(range(0, len(clips), 64), desc=f"clean {name}"):
        xs = [clips.get(i) for i in range(i0, min(len(clips), i0 + 64))]
        L = max(len(x) for x in xs)
        wav = torch.from_numpy(np.stack([np.pad(x, (0, L - len(x))) for x in xs])).to(dev)
        wav = wav * (0.05 / wav.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-6))   # same level as in training
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            P = torch.sigmoid(m_old(feats(wav)).float()).cpu().numpy()          # (B, tokens, old classes)
        for j, x in enumerate(xs):
            n_tok = int(len(x) / SR / FRAME)
            hit = (P[j, :n_tok][:, use] > old_thr[use]).any(1)
            hit = np.convolve(hit, np.ones(17), "same") > 0                     # +-0.32 s around each hit
            flagged += hit.mean()
            best, run = (0, 0), None
            for t, h in enumerate(np.append(hit, True)):
                if not h and run is None: run = t
                if h and run is not None:
                    if t - run > best[1] - best[0]: best = (run, t)
                    run = None
            a, b = int(best[0] * FRAME * SR), min(len(x), int(best[1] * FRAME * SR))
            if b - a >= 2 * SR: segs.append([i0 + j, a, b])
    json.dump(segs, open(out, "w"))
    print(f"{name}: kept {len(segs)}/{len(clips)} | {100*flagged/max(1, len(clips)):.0f}% of the audio had detected events")
    return segs

beds_ok = clean_segments(beds, "beds", vocal_only=True)
noise_ok = clean_segments(noise_bank, "noise", vocal_only=False)
if BACKUP_CACHE: backup(BANK)
''')

# ----------------------------------------------------------------------------------------------- datasets
md(r'''
## Step 4: Training examples and augmentation
- `MixDataset`: a clean speech bed plus 0–3 randomly placed event crops from all four datasets, with classes sampled uniformly. The timestamps are exact because the mixer places the events.
- `RealWindows`: real clips from one source (NV38k, Vaani or NonverbalTTS) joined end to end into 8 s windows, keeping their annotated timestamps.
- `Augmented`: every example is emitted `1 + copies` times. The first copy is clean. Each other copy is a different real-world version:

| augmentation | probability | what it mimics |
|---|---|---|
| silence inserted at 1–2 points outside events (0.2–1.5 s), labels shifted | 0.5 | pauses, gaps, start/end silence |
| reverb (synthetic room, RT60 0.15–0.8 s) | 0.25 | rooms, distant mic |
| phone band 300–3400 Hz + μ-law | 0.2 | phone and call-centre audio |
| background noise: **real recordings (DNC, DEMAND)** 40%, white/pink/brown 30%, mains hum 10%, babble 20% | 0.85 | street, home, office, café, transport, fan or AC, electrical hum, crowd |
| gain −12…+6 dB | 1 | loud and quiet recordings |

With a `seed`, item *i* is always the same, so validation sets are fixed. Unlabelled hum and noise also teach the model *not* to fire `machine`/`fan` on steady hum, the main source of false tags before.
''')

code(r'''
def rms(x): return float(np.sqrt(np.mean(x ** 2)) + 1e-6)
def speed(x, r):     # cheap speed/pitch jitter (torchaudio resample at odd ratios is very slow on CPU)
    n = max(1, int(len(x) / r)); return np.interp(np.arange(n) * r, np.arange(len(x)), x).astype(np.float32)

class MixDataset(Dataset):
    def __init__(self, n, split, seed=None, p_no_speech=0.10, p_no_event=0.15):
        self.n, self.seed, self.p_no_speech, self.p_no_event = n, seed, p_no_speech, p_no_event
        by = collections.defaultdict(list)
        for i, m in enumerate(ev_bank.meta):
            if m["split"] == split and m["label"] in cls2idx: by[m["label"]].append(i)
        if not by:   # tiny smoke runs may have no val crops; fall back to train crops
            for i, m in enumerate(ev_bank.meta): by[m["label"]].append(i)
        self.by, self.labels = dict(by), sorted(by)
        self.beds = [s for s in beds_ok if beds.meta[s[0]]["split"] == split] or beds_ok

    def __len__(self): return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else (self.seed, i))
        mix, evs, ref = np.zeros(T_SAMP, np.float32), [], 0.05
        has_speech = bool(self.beds) and rng.random() > self.p_no_speech
        if has_speech:
            b, a, z = self.beds[rng.integers(len(self.beds))]
            s = beds.get(b)[a:z]
            if len(s) < T_SAMP: s = np.tile(s, math.ceil(T_SAMP / len(s)))[:T_SAMP]
            else: o = rng.integers(0, len(s) - T_SAMP + 1); s = s[o: o + T_SAMP]
            mix += s * (ref / rms(s))
        if rng.random() > self.p_no_event:
            for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
                lab = self.labels[rng.integers(len(self.labels))]          # uniform over classes = balanced
                x = ev_bank.get(self.by[lab][rng.integers(len(self.by[lab]))])
                if rng.random() < 0.3: x = speed(x, rng.uniform(0.9, 1.1))
                if len(x) > T_SAMP: o = rng.integers(0, len(x) - T_SAMP + 1); x = x[o: o + T_SAMP]
                d = len(x) / SR
                t0 = rng.uniform(0, max(0.0, DUR - d))
                snr = rng.uniform(-3, 12) if has_speech else rng.uniform(10, 25)   # speech-to-event dB
                i0 = int(t0 * SR)
                mix[i0: i0 + len(x)] += x * (ref / rms(x) * 10 ** (-snr / 20))
                evs.append((cls2idx[lab], round(t0, 3), round(t0 + d, 3)))
        mix *= 10 ** (rng.uniform(-10, 6) / 20)                             # overall level jitter
        mix += rng.standard_normal(T_SAMP).astype(np.float32) * rng.uniform(1e-4, 1e-3)
        return mix.astype(np.float32), evs

class RealWindows(Dataset):
    # Real clips of the given sources joined into DUR-second windows; a source is picked uniformly, then a clip.
    def __init__(self, n, split, srcs, seed=None):
        self.n, self.seed = n, seed
        by = collections.defaultdict(list)
        for i, m in enumerate(real.meta):
            if m["split"] == split and m["src"] in srcs: by[m["src"]].append(i)
        self.by = [v for v in by.values() if v]
    def __len__(self): return self.n if self.by else 0
    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else (self.seed, i))
        out, evs, t, first = np.zeros(T_SAMP, np.float32), [], 0, True
        fade = np.linspace(0, 1, int(0.01 * SR), dtype=np.float32)
        while t < T_SAMP:
            pool = self.by[rng.integers(len(self.by))]
            j = pool[rng.integers(len(pool))]
            x = real.get(j)
            off = int(rng.integers(0, max(1, len(x) - SR))) if first else 0   # start part-way into the 1st clip
            if T_SAMP - t < 2 * len(fade): break
            seg = x[off: off + T_SAMP - t].copy()
            if len(seg) < 2 * len(fade): continue
            seg *= 0.05 * 10 ** (rng.uniform(-6, 6) / 20) / rms(seg)
            seg[: len(fade)] *= fade; seg[-len(fade):] *= fade[::-1]
            out[t: t + len(seg)] = seg
            for c, s, e in real.meta[j]["events"]:
                s2, e2 = max(0.0, (s * SR - off + t) / SR), min((t + len(seg)) / SR, (e * SR - off + t) / SR)
                if e2 - s2 >= 0.05: evs.append((cls2idx[c], round(s2, 3), round(e2, 3)))
            t += len(seg); first = False
        return out, evs

# ---- augmentation: noise, silence, reverb, phone band, gain ----
PHONE_SOS = scipy.signal.butter(4, [300, 3400], "bandpass", fs=SR, output="sos")

def colored_noise(n, rng, beta):            # beta 0 white, 1 pink, 2 brown
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / f ** (beta / 2)
    x = np.fft.irfft(spec, n).astype(np.float32); return x / rms(x)

def hum(n, rng):                            # mains hum with harmonics, fan/AC-like
    t = np.arange(n) / SR; f0 = rng.choice([50.0, 60.0]) * rng.uniform(0.99, 1.01)
    x = sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 8))
    x = x.astype(np.float32) + 0.3 * colored_noise(n, rng, 1); return x / rms(x)

def babble(n, rng):                         # 3-6 overlapping talkers from the speech beds
    x = np.zeros(n, np.float32)
    for _ in range(rng.integers(3, 7)):
        b, a, z = beds_ok[rng.integers(len(beds_ok))]
        s = beds.get(b)[a:z]; s = np.tile(s, math.ceil(n / len(s)))[:n]
        x += np.roll(s, rng.integers(n)) / rms(s)
    return x / rms(x)

def env_noise(n, rng):                     # a real recorded background (DNC quiet/people, DEMAND)
    b, a, z = noise_ok[rng.integers(len(noise_ok))]
    s = noise_bank.get(b)[a:z]; s = np.tile(s, math.ceil(n / len(s)))[:n]
    return np.roll(s, rng.integers(n)) / rms(s)

def reverb(x, rng):
    rt60 = rng.uniform(0.15, 0.8); L = int(rt60 * SR)
    ir = rng.standard_normal(L).astype(np.float32) * np.exp(-6.9 * np.arange(L) / L).astype(np.float32)
    ir[0] = rng.uniform(1, 4) * np.abs(ir).max()                  # direct path
    y = scipy.signal.fftconvolve(x, ir)[: len(x)].astype(np.float32)
    return y * (rms(x) / rms(y))

def phone(x, rng):
    y = scipy.signal.sosfilt(PHONE_SOS, x).astype(np.float32)
    if rng.random() < 0.5:                                        # mu-law 8-bit codec
        mu = 255.0; pk = np.abs(y).max() + 1e-6; z = y / pk
        q = np.round(np.sign(z) * np.log1p(mu * np.abs(z)) / np.log1p(mu) * 127) / 127
        y = (np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu * pk).astype(np.float32)
    return y * (rms(x) / rms(y))

def insert_silence(x, evs, rng):
    # Insert 1-2 pauses (0.2-1.5 s) at points outside every event; events after a pause move right.
    for _ in range(rng.integers(1, 3)):
        for _try in range(10):
            t = rng.uniform(0, DUR - 0.5)
            if not any(s - 0.05 < t < e + 0.05 for _, s, e in evs): break
        else:
            break
        g = rng.uniform(0.2, 1.5); i = int(t * SR)
        x = np.concatenate([x[:i], np.zeros(int(g * SR), np.float32), x[i:]])[:T_SAMP]
        evs = [(c, s + g, e + g) if s >= t else (c, s, e) for c, s, e in evs]
        evs = [(c, round(s, 3), round(min(e, DUR), 3)) for c, s, e in evs if min(e, DUR) - s >= 0.05]
    return x, evs

def augment(x, evs, rng):
    x, evs = x.copy(), list(evs)
    if rng.random() < 0.5: x, evs = insert_silence(x, evs, rng)
    if rng.random() < 0.25: x = reverb(x, rng)
    if rng.random() < 0.2: x = phone(x, rng)
    if rng.random() < 0.85:
        kind = rng.choice(["env", "white", "pink", "brown", "hum", "babble"], p=[0.4, 0.08, 0.12, 0.1, 0.1, 0.2])
        if kind == "env" and noise_ok: n, snr = env_noise(T_SAMP, rng), rng.uniform(3, 25)
        elif kind == "babble" and beds_ok: n, snr = babble(T_SAMP, rng), rng.uniform(12, 30)
        elif kind == "hum": n, snr = hum(T_SAMP, rng), rng.uniform(10, 35)
        else: n, snr = colored_noise(T_SAMP, rng, {"white": 0, "pink": 1, "brown": 2}.get(kind, 1)), rng.uniform(8, 35)
        x = x + n * rms(x) * 10 ** (-snr / 20)
    x *= 10 ** (rng.uniform(-12, 6) / 20)
    return np.clip(x, -1, 1).astype(np.float32), evs

class Augmented(Dataset):
    # Each base example i -> (1 + copies) examples: the clean original, then `copies` different augmentations.
    def __init__(self, ds, copies, seed, keep_clean=True):
        self.ds, self.k, self.seed, self.keep_clean = ds, copies + int(keep_clean), seed, keep_clean
    def __len__(self): return len(self.ds) * self.k
    def __getitem__(self, i):
        x, evs = self.ds[i // self.k]
        v = i % self.k
        if self.keep_clean and v == 0: return x, evs
        return augment(x, evs, np.random.default_rng((self.seed, i)))

def collate(b):
    return torch.from_numpy(np.stack([w for w, _ in b])), [e for _, e in b]

# Sanity check: listen to a few examples (clean and augmented) before spending GPU time
os.makedirs(f"{RUN}/debug", exist_ok=True)
for name, ds in [("mix", Augmented(MixDataset(2, "train", seed=0), 1, seed=5)),
                 *[(f"real_{s}", Augmented(RealWindows(1, "train", [s], seed=0), 2, seed=5)) for s in SRCS]]:
    for i in range(len(ds)):
        x, evs = ds[i]
        sf.write(f"{RUN}/debug/{name}_{i}.wav", x, SR)
        print(f"{name}_{i}", "(clean)" if i % ds.k == 0 else "(augmented)", [(classes[c], s, e) for c, s, e in evs])
''')

code(r'''
from IPython.display import Audio, display
for f in sorted(glob.glob(f"{RUN}/debug/real_nvtts_*.wav"))[:3]:
    print(os.path.basename(f)); display(Audio(f))
''')

# ----------------------------------------------------------------------------------------------- features
md(r'''
## Step 5: Feature cache (log-mel fp16 on disk)
Fixed pools are generated once and turned into log-mel on the GPU. They're stored as fp16 memmaps of about 100 KB per 8 s example. Training then only reads features, so the CPUs never hold the GPU back. On top of the stored augmentation, each batch gets SpecAugment and a random circular time shift. Validation sets are clean real windows per source, plus one augmented ("noisy") set that checks robustness.
''')

code(r'''
FEAT = os.path.join(RUN, "feats"); os.makedirs(FEAT, exist_ok=True)

class FeatCache(Dataset):
    def __init__(self, name):
        self.evs = json.load(open(f"{FEAT}/{name}.json"))
        self.mm = (np.memmap(f"{FEAT}/{name}.f16", np.float16, "r", shape=(len(self.evs), N_MELS, N_FR))
                   if self.evs else np.zeros((0, N_MELS, N_FR), np.float16))
    def __len__(self): return len(self.evs)
    def __getitem__(self, i): return np.asarray(self.mm[i], np.float32), [tuple(e) for e in self.evs[i]]

@torch.no_grad()
def build_feats(name, ds):
    if os.path.exists(f"{FEAT}/{name}.json"):
        return FeatCache(name)
    n = len(ds)
    mm = np.memmap(f"{FEAT}/{name}.f16", np.float16, "w+", shape=(max(n, 1), N_MELS, N_FR))
    evs, i = [], 0
    if n:
        dl = DataLoader(ds, batch_size=64, num_workers=CFG["workers"], collate_fn=collate)
        for wav, e in tqdm(dl, desc=f"features:{name}"):
            mm[i: i + len(wav)] = feats(wav.to(dev)).half().cpu().numpy(); evs += e; i += len(wav)
    mm.flush(); del mm
    json.dump(evs, open(f"{FEAT}/{name}.json", "w"))
    return FeatCache(name)

tr_synth = build_feats("train_synth", Augmented(MixDataset(CFG["pool_synth"], "train", seed=1), CFG["aug_copies_synth"], seed=11))
tr_real = {s: build_feats(f"train_real_{s}", Augmented(RealWindows(CFG["pool_real"][s], "train", [s], seed=2 + k),
                                                       CFG["aug_copies_real"], seed=21 + k))
           for k, s in enumerate(SRCS)}
va_synth = build_feats("val_synth", MixDataset(CFG["val_synth"], "val", seed=123))
va_real = {s: build_feats(f"val_real_{s}", RealWindows(CFG["val_real"][s], "val", [s], seed=456 + k))
           for k, s in enumerate(SRCS)}
va_noisy = build_feats("val_noisy", Augmented(RealWindows(CFG["val_noisy"], "val", SRCS, seed=789), 1, seed=790, keep_clean=False))
for n, d in [("train_synth", tr_synth), *[(f"train_real_{s}", d) for s, d in tr_real.items()],
             ("val_synth", va_synth), *[(f"val_real_{s}", d) for s, d in va_real.items()], ("val_noisy", va_noisy)]:
    print(f"{n:18s} {len(d):7d} examples, {d.mm.nbytes/1e9 if len(d) else 0:.2f} GB")
if BACKUP_CACHE: backup(FEAT)
print("disk free:", f"{shutil.disk_usage(FEAT).free/1e9:.0f} GB |", mem())
''')

# ----------------------------------------------------------------------------------------------- model
md(r'''
## Step 6: Model and warm start
With `INIT_FROM`, the encoder starts from the current model. Head rows are copied for the classes it already knows, and new classes (e.g. `groan`) start fresh.
''')

code(r'''
def spec_augment(mel, t_max=30, f_max=8):     # on GPU, per example
    B, M, T = mel.shape
    t0 = torch.randint(0, T - t_max, (B, 1, 1), device=mel.device); tw = torch.randint(1, t_max, (B, 1, 1), device=mel.device)
    f0 = torch.randint(0, M - f_max, (B, 1, 1), device=mel.device); fw = torch.randint(1, f_max, (B, 1, 1), device=mel.device)
    ti = torch.arange(T, device=mel.device).view(1, 1, T); fi = torch.arange(M, device=mel.device).view(1, M, 1)
    mask = ((ti >= t0) & (ti < t0 + tw)) | ((fi >= f0) & (fi < f0 + fw))
    return mel.masked_fill(mask, 0.0)

model = SED(C).to(dev)
if INIT_FROM and os.path.exists(INIT_FROM):
    ck = torch.load(INIT_FROM, map_location=dev, weights_only=True)
    model.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith("head.")}, strict=False)
    with torch.no_grad():
        for c in classes:
            if c in ck["classes"]:
                j = ck["classes"].index(c)
                model.head.weight[cls2idx[c]] = ck["model"]["head.weight"][j]
                model.head.bias[cls2idx[c]] = ck["model"]["head.bias"][j]
    print(f"warm start from {INIT_FROM}; new classes: {[c for c in classes if c not in ck['classes']]}")
else:
    print("training from scratch")
with torch.no_grad():
    L_TOK = model(torch.zeros(1, N_MELS, N_FR, device=dev)).shape[1]
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {C} classes, {L_TOK} tokens per {DUR:.0f}s window")
''')

# ----------------------------------------------------------------------------------------------- train
md(r'''
## Step 7: Training (reads cached features only)
The checkpoint is chosen by **macro average precision on the real held-out sets** (NV38k, Vaani, NonverbalTTS and noisy), not by F1 at a fixed 0.5 threshold. AP doesn't depend on a threshold, so it rewards a model that ranks real events above everything else. Recall can then be bought with a lower threshold in the next step. Each class is scored only on the datasets that label it. The run resumes from `last.pt` after an interruption.
''')

code(r'''
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

def frame_targets(evs_list, L, dilate=1):
    y = torch.zeros(len(evs_list), L, C)
    for b, evs in enumerate(evs_list):
        for c, s, e in evs:
            y[b, max(0, int(s / FRAME) - dilate): min(L, int(e / FRAME) + 1 + dilate), int(c)] = 1.
    return y

train_parts = [d for d in (tr_synth, *tr_real.values()) if len(d)]
all_evs = [e for d in train_parts for e in d.evs]
pos = sum(frame_targets(all_evs[i: i + 2000], L_TOK).sum((0, 1)) for i in range(0, len(all_evs), 2000))
tot = len(all_evs) * L_TOK
pw = ((tot - pos) / pos.clamp(min=1)).clamp(1, 100).to(dev)       # class-balanced pos_weight

train_ds = ConcatDataset(train_parts)
loader = DataLoader(train_ds, batch_size=CFG["bs"], shuffle=True, drop_last=True, collate_fn=collate,
                    num_workers=CFG["workers"], pin_memory=True, persistent_workers=True)
val_sets = {**{f"real_{s}": d for s, d in va_real.items() if len(d)}, "noisy": va_noisy, "synth": va_synth}
val_sets = {k: v for k, v in val_sets.items() if len(v)}
SCORED = [k for k in val_sets if k != "synth"]                    # model selection uses real audio only
SRC_OF = {f"real_{s}": s for s in SRCS}
opt = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-2)
sched = OneCycleLR(opt, max_lr=CFG["lr"], total_steps=len(loader) * CFG["epochs"], pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
CK = os.path.join(RUN, "ckpt"); os.makedirs(CK, exist_ok=True)
print(f"{len(train_ds)} training examples, {len(loader)} steps/epoch | val: {({k: len(v) for k, v in val_sets.items()})}")

@torch.no_grad()
def predict(ds, bs=64, m=None):
    m = m or model; m.eval(); P, Y = [], []
    for mel, evs in DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=min(4, CFG["workers"])):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            P.append(torch.sigmoid(m(mel.to(dev)).float()).cpu())
        Y.append(frame_targets(evs, P[-1].size(1)))
    model.train()
    return torch.cat(P), torch.cat(Y)

def average_precision(p, y):                    # 1-D tensors; frame-level AP
    o = torch.argsort(p, descending=True); y = y[o].float()
    tp = torch.cumsum(y, 0)
    return float((tp / torch.arange(1, len(y) + 1) * y).sum() / y.sum().clamp(min=1))

def class_mask(name):                           # classes labelled by the source behind a val set
    s = SRC_OF.get(name)
    return np.array([s is None or c in LABELS[s] for c in classes])

def macro_ap(P, Y, mask):
    aps = {classes[c]: average_precision(P[..., c].flatten(), Y[..., c].flatten())
           for c in range(C) if mask[c] and Y[..., c].sum() > 0}
    return float(np.mean(list(aps.values()))) if aps else float("nan"), aps

start, best = 0, -1.0
if os.path.exists(f"{CK}/last.pt"):            # resume after an interruption
    s = torch.load(f"{CK}/last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    scaler.load_state_dict(s["scaler"]); start, best = s["epoch"] + 1, s["best"]
    print("resumed at epoch", start)

for ep in range(start, CFG["epochs"]):
    t0, run_loss = time.time(), 0.0
    for step, (mel, evs) in enumerate(tqdm(loader, desc=f"epoch {ep}", leave=False)):
        mel = mel.to(dev, non_blocking=True)
        y = frame_targets(evs, L_TOK).to(dev)
        k = int(np.random.randint(0, L_TOK))                       # circular shift: mel by 4k frames, labels by k
        mel, y = torch.roll(mel, 4 * k, dims=2), torch.roll(y, k, dims=1)
        if np.random.rand() < 0.5: mel = spec_augment(mel)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logits = model(mel)
        logits = logits.float()[:, :L_TOK]
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        loss = loss + 0.25 * F.binary_cross_entropy_with_logits(logits.amax(1), y.amax(1))   # clip-level aux
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        run_loss += loss.item()
    aps = {n: macro_ap(*predict(d), class_mask(n))[0] for n, d in val_sets.items()}
    score = float(np.nanmean([aps[n] for n in SCORED])) if SCORED else aps.get("synth", 0.0)
    print(f"epoch {ep}: loss {run_loss/max(1, step+1):.4f}  macro-AP " + "  ".join(f"{n} {v:.3f}" for n, v in aps.items())
          + f"  | score {score:.3f}  {time.time()-t0:.0f}s  {mem()}")
    if score > best:
        best = score; torch.save({"model": model.state_dict(), "classes": classes}, f"{CK}/best.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep, "best": best}, f"{CK}/last.pt")
    backup(CK)                                  # Colab: copy checkpoints to Drive every epoch
''')

# ----------------------------------------------------------------------------------------------- thresholds
md(r'''
### Per-class thresholds: two profiles, cross-validated on real held-out audio
Each class is tuned on the real validation sets of the datasets that label it, with frame-level scores.
- `thresholds.json` (**balanced**): best F1. Weak classes are floored at 0.9.
- `thresholds_sensitive.json` (**sensitive**, the default in `scripts/infer.py`): best F2, so a missed sound counts twice as much as a false tag. Weak classes are clamped to 0.6–0.8.

The windows are split in two halves: tune on one, score on the other, then swap. The CV columns are therefore honest. The final thresholds are tuned on all windows. A class is "weak" if it has fewer than 25 positive frames (1 s) or a cross-validated F1 below 0.1.
''')

code(r'''
model.load_state_dict(torch.load(f"{CK}/best.pt", map_location=dev, weights_only=True)["model"])
preds = {n: predict(d) for n, d in val_sets.items() if n.startswith("real_")}
GRID = np.arange(0.05, 0.99, 0.025)

def prf(p, y, t):
    pr, gt = p > t, y > .5
    tp, fp, fn = (pr & gt).sum().item(), (pr & ~gt).sum().item(), (~pr & gt).sum().item()
    P_, R_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f = lambda b: (1 + b * b) * P_ * R_ / max(1e-9, b * b * P_ + R_)
    return P_, R_, f(1), f(2)

def tune(p, y, beta_i):                         # beta_i: 2 -> F1, 3 -> F2 (index into prf output)
    return max(GRID, key=lambda t: prf(p, y, t)[beta_i])

rows, thr_bal, thr_sens = {}, {}, {}
for c, name in enumerate(classes):
    srcs = [n for n in preds if name in LABELS[SRC_OF[n]]]
    if not srcs:
        thr_bal[name], thr_sens[name] = 0.9, 0.8; continue
    p = torch.cat([preds[n][0][..., c] for n in srcs]); y = torch.cat([preds[n][1][..., c] for n in srcs])
    halves = [(slice(0, None, 2), slice(1, None, 2)), (slice(1, None, 2), slice(0, None, 2))]
    cv = {}
    for bi, key in ((2, "F1"), (3, "F2")):     # 2-fold CV: tune on one half of the windows, score on the other
        sc = [prf(p[b], y[b], tune(p[a], y[a], bi)) for a, b in halves]
        cv[key] = np.mean([s[bi] for s in sc]); cv[key + "_P"] = np.mean([s[0] for s in sc]); cv[key + "_R"] = np.mean([s[1] for s in sc])
    n_pos = int((y > .5).sum())
    weak = n_pos < 25 or cv["F1"] < 0.1
    tb, ts = float(tune(p, y, 2)), float(tune(p, y, 3))
    thr_bal[name] = round(max(tb, 0.9) if weak else tb, 3)
    thr_sens[name] = round(min(max(ts, 0.6), 0.8) if weak else ts, 3)
    rows[name] = dict(eval_on="+".join(SRC_OF[n] for n in srcs), pos_frames=n_pos,
                      AP=round(average_precision(p.flatten(), y.flatten()), 3),
                      thr_bal=thr_bal[name], cv_F1=round(cv["F1"], 3), P=round(cv["F1_P"], 3), R=round(cv["F1_R"], 3),
                      thr_sens=thr_sens[name], cv_F2=round(cv["F2"], 3), P_sens=round(cv["F2_P"], 3), R_sens=round(cv["F2_R"], 3),
                      note="weak" if weak else "")
report = pd.DataFrame(rows).T.sort_values("pos_frames", ascending=False)
print(report.to_string())
ok = report[report.note == ""]
print(f"\nreliable classes: {len(ok)} | mean CV recall: balanced {ok.R.mean():.3f}, sensitive {ok.R_sens.mean():.3f} "
      f"| mean CV precision: balanced {ok.P.mean():.3f}, sensitive {ok.P_sens.mean():.3f}")
json.dump(thr_bal, open(f"{CK}/thresholds.json", "w"), indent=1)
json.dump(thr_sens, open(f"{CK}/thresholds_sensitive.json", "w"), indent=1)
json.dump({"low_ratio": 0.6}, open(f"{CK}/detect_config.json", "w"))
report.to_csv(f"{CK}/threshold_report.csv")
backup(CK)
''')

# ----------------------------------------------------------------------------------------------- inference
md("## Step 8: Inference (waveform → events with timestamps)")

code(r'''
from scipy.ndimage import median_filter, find_objects, label as label_runs

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

class Detector:
    def __init__(self, ckpt, thresholds, device=dev, low_ratio=0.6):
        ck = torch.load(ckpt, map_location=device, weights_only=True)
        self.classes, self.dev, self.low_ratio = ck["classes"], device, low_ratio
        self.model = SED(len(self.classes)).to(device).eval(); self.model.load_state_dict(ck["model"])
        self.feats = LogMel().to(device)
        self.thr = np.array([thresholds.get(c, 0.5) for c in self.classes])

    @classmethod
    def from_dir(cls, d, profile="sensitive"):
        thr = json.load(open(f"{d}/{'thresholds_sensitive.json' if profile == 'sensitive' else 'thresholds.json'}"))
        cfg = json.load(open(f"{d}/detect_config.json")) if os.path.exists(f"{d}/detect_config.json") else {}
        return cls(f"{d}/best.pt", thr, low_ratio=thr.pop("_low_ratio", cfg.get("low_ratio", 0.6)))

    @torch.no_grad()
    def frame_probs(self, x, win_s=DUR, hop_s=2.0):        # x: 1-D float array/tensor at 16 kHz
        x = torch.as_tensor(x, dtype=torch.float32)
        win, hop = int(win_s * SR), int(hop_s * SR)
        n_tok = int(len(x) / SR / FRAME) + 1
        acc, cnt = np.zeros((n_tok, len(self.classes))), np.zeros(n_tok)
        for t0 in range(0, max(1, len(x) - win + hop), hop):
            seg = x[t0: t0 + win]
            seg = F.pad(seg, (0, win - len(seg)))
            p = torch.sigmoid(self.model(self.feats(seg[None].to(self.dev))))[0].float().cpu().numpy()
            g = t0 // int(FRAME * SR) + np.arange(len(p)); ok = g < n_tok
            np.add.at(acc, g[ok], p[ok]); np.add.at(cnt, g[ok], 1)
        acc /= np.maximum(cnt, 1)[:, None]
        return np.arange(n_tok) * FRAME, acc

    def detect(self, audio, min_dur=0.15, gap_inline=0.30, gap_ambient=1.0, low_ratio=None):
        low_ratio = self.low_ratio if low_ratio is None else low_ratio
        if isinstance(audio, str):
            w, sr = sf.read(audio, dtype="float32", always_2d=True)
            audio = torchaudio.functional.resample(torch.from_numpy(w.mean(1)), sr, SR) if sr != SR else w.mean(1)
        times, probs = self.frame_probs(audio)
        probs = median_filter(probs, size=(5, 1))                     # de-jitter
        events = []
        for ci, name in enumerate(self.classes):
            gap = gap_inline if name in INLINE else gap_ambient
            for s, e in class_runs(probs[:, ci], self.thr[ci], low_ratio, gap, min_dur):
                events.append(dict(label=name, start=round(float(s * FRAME), 2), end=round(float(e * FRAME), 2),
                                   prob=round(float(probs[s:e, ci].mean()), 2)))
        return sorted(events, key=lambda e: e["start"])

detector = Detector(f"{CK}/best.pt", json.load(open(f"{CK}/thresholds_sensitive.json")))
val_clips = {s: [i for i, m in enumerate(real.meta) if m["split"] == "val" and m["src"] == s] for s in SRCS}
for s, ids in val_clips.items():
    if not ids: continue
    j = ids[0]
    print(f"[{s}] ground truth:", real.meta[j]["events"])
    print(f"[{s}] detected:    ", detector.detect(real.get(j)))
sf.write(f"{RUN}/test.wav", real.get(next(ids for ids in val_clips.values() if ids)[0]), SR)
''')

# ----------------------------------------------------------------------------------------------- eval
md(r'''
## Step 9: Event-level evaluation on whole held-out clips
The clips are held-out real recordings: Vaani from unseen districts with verified timestamps, NV38k from the held-out hash split, NonverbalTTS dev+test and the AudioSet-Strong evaluation split. If a model is present in `checkpoints/`, it scores the same clips (`current_vs_new.csv`). A true event counts as **found** if a same-label prediction overlaps it, and a prediction counts as **correct** if it overlaps a same-label true event. Each class is scored only on datasets that label it. The current model uses its `sensitive` thresholds, and the new model uses both of its profiles.

Thresholds were tuned on held-out data that overlaps these clips, so the absolute numbers are a little optimistic.
''')

code(r'''
def overlap_eval(det, clip_ids):
    HG, NG, HP, NP = (collections.Counter() for _ in range(4))
    for s, ids in clip_ids.items():
        lab = LABELS[s]
        for j in tqdm(ids, desc=s, leave=False):
            gt = real.meta[j]["events"]
            pred = [p for p in det.detect(real.get(j)) if p["label"] in lab]
            for c, a, b in gt:
                NG[c] += 1; HG[c] += any(p["label"] == c and p["start"] < b and p["end"] > a for p in pred)
            for p in pred:
                NP[p["label"]] += 1; HP[p["label"]] += any(c == p["label"] and p["start"] < b and p["end"] > a for c, a, b in gt)
    df = pd.DataFrame({c: dict(gt=NG[c], recall=HG[c] / max(1, NG[c]), pred=NP[c], precision=HP[c] / max(1, NP[c]))
                       for c in set(NG) | set(NP)}).T.astype(float)
    df["F1"] = (2 * df.precision * df.recall / (df.precision + df.recall).replace(0, np.nan)).fillna(0)
    return df

eval_ids = {s: ids[: CFG["eval_clips"]] for s, ids in val_clips.items() if ids}
models = {"new_sensitive": detector, "new_balanced": Detector(f"{CK}/best.pt", json.load(open(f"{CK}/thresholds.json")))}
if os.path.exists(f"{OLD_DIR}/thresholds_sensitive.json") and os.path.exists(f"{OLD_DIR}/best.pt"):
    models = {"current_sensitive": Detector.from_dir(OLD_DIR, "sensitive"), **models}
res = {k: overlap_eval(d, eval_ids) for k, d in models.items()}
cmp = pd.concat({k: v[["recall", "precision", "F1"]] for k, v in res.items()}, axis=1)
cmp.insert(0, "gt_events", res["new_sensitive"]["gt"])
cmp = cmp.sort_values("gt_events", ascending=False).round(3)
print(cmp.to_string())
common = cmp[cmp.gt_events >= 10].index
print("\nmacro over classes with >= 10 held-out events:")
print(pd.DataFrame({k: v.loc[v.index.intersection(common), ["recall", "precision", "F1"]].mean() for k, v in res.items()}).round(3).to_string())
ev = pd.DataFrame({"events": res["new_sensitive"]["gt"].astype(int)})          # same format as the released event_eval.csv
for prof in ("sensitive", "balanced"):
    for m_ in ("recall", "precision", "F1"): ev[f"{prof}_{m_}"] = res[f"new_{prof}"][m_].round(3)
ev.index.name = "class"; ev.sort_values("events", ascending=False).to_csv(f"{CK}/event_eval.csv")
if "current_sensitive" in res: cmp.to_csv(f"{CK}/current_vs_new.csv")
''')

# ----------------------------------------------------------------------------------------------- export
md(r'''
## Step 10: Export
This writes the model (`best.pt`, plus `model.safetensors` + `config.json` for the Hub), both threshold profiles, the detector config and the reports to `EXPORT`, in the same layout as `checkpoints/`. Use it with `python scripts/infer.py audio.wav --model checkpoints_new`. Once it's better, copy it over `checkpoints/` by hand. Nothing here overwrites the shipped model.
''')

code(r'''
os.makedirs(EXPORT, exist_ok=True)
ck_best = torch.load(f"{CK}/best.pt", map_location="cpu", weights_only=True)
torch.save(ck_best, f"{EXPORT}/best.pt")
from safetensors.torch import save_file                      # Hub format: weights without pickle
save_file({k: v.contiguous() for k, v in ck_best["model"].items()}, f"{EXPORT}/model.safetensors", metadata={"format": "pt"})
json.dump(dict(model_type="sed-cnn-transformer", classes=classes, inline_classes=[c for c in classes if c in INLINE],
               background_classes=[c for c in classes if c not in INLINE], sample_rate=SR, n_mels=N_MELS,
               frame_hop_s=FRAME, window_s=DUR, inference_hop_s=2.0,
               threshold_profiles={"sensitive": "thresholds_sensitive.json", "balanced": "thresholds.json"}),
          open(f"{EXPORT}/config.json", "w"), indent=1)
for f in ("thresholds.json", "thresholds_sensitive.json", "detect_config.json", "threshold_report.csv", "event_eval.csv", "current_vs_new.csv"):
    if os.path.exists(f"{CK}/{f}"): shutil.copy(f"{CK}/{f}", f"{EXPORT}/{f}")
json.dump(dict(classes=classes, cfg={k: v for k, v in CFG.items()}, smoke=SMOKE, init_from=INIT_FROM,
               data=[NV_REPO, VA_REPO if HAVE_VAANI else None, TTS_REPO, AS_REPO,
                     "OpenSLR 99 (Deeply Nonverbal Vocalization)", "DNC (TU Berlin)", "DEMAND (Zenodo 1227121)"]),
          open(f"{EXPORT}/train_info.json", "w"), indent=1, default=str)
print(EXPORT, sorted(os.listdir(EXPORT)))
''')

code(r'''
# Download the exported model from Colab (or copy it to Drive by setting EVENT_CACHE to a Drive folder)
shutil.make_archive("/content/checkpoints_new", "zip", EXPORT)
from google.colab import files
files.download("/content/checkpoints_new.zip")
''', "colab")

# ----------------------------------------------------------------------------------------------- whisper
md(r'''
## Step 11: Merge with a transcription engine
Speaker sounds (`INLINE`) are written inline as `[tag]`, and background sounds become `<tag> ... </tag>`. `clean()` drops an event the ASR already wrote as text, such as a laugh transcribed as "haha".
''')

code(r'''
import bisect
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

demo_words = [("Hey!", 0.2, 0.6), ("how", 1.6, 1.8), ("are", 1.8, 1.9), ("you?", 1.9, 2.2)]
demo_events = [dict(label="gasp", start=1.0, end=1.4, prob=.9), dict(label="horn", start=1.7, end=1.9, prob=.8)]
print(rich_transcript(demo_words, clean(demo_words, demo_events)))
''')

code(r'''
# Whisper + event detector on your own audio (any format ffmpeg reads: wav, mp3, m4a, ogg...)
%pip -q install -U openai-whisper
import whisper

ASR_MODEL = "small"     # multilingual; "small.en" for English-only audio, "medium" for better text (slower)
asr = whisper.load_model(ASR_MODEL, device=dev, download_root=os.path.join(os.environ.get("TORCH_HOME", CACHE), "whisper"))

HIDE        = set()                    # e.g. {"breath", "lip_smack"} to leave those tags out
DETECT_KW   = dict(low_ratio=0.6, gap_inline=0.30, gap_ambient=1.0, min_dur=0.15)
TRANSCR_KW  = dict(join_inline=0.5, join_ambient=1.5)

def transcribe_with_events(path, language=None):
    # Runs Whisper and the event detector once; returns (words, events) for rendering.
    audio = whisper.load_audio(path)                         # ffmpeg -> 16 kHz mono float32
    r = asr.transcribe(audio, word_timestamps=True, language=language, fp16=dev == "cuda")
    words = [(w["word"], w["start"], w["end"]) for s in r["segments"] for w in s.get("words", [])]
    events = [e for e in detector.detect(audio, **DETECT_KW) if e["label"] not in HIDE]
    return words, clean(words, events)

# Put the path of your own file here (or, on Colab, uncomment the upload lines).
# from google.colab import files
# AUDIO = "/content/" + list(files.upload())[0]
AUDIO = f"{RUN}/test.wav"                                    # a held-out clip from Step 8

words, events = transcribe_with_events(AUDIO)
print(rich_transcript(words, events, **TRANSCR_KW), "\n")
print(rich_transcript(words, events, include_times=True, **TRANSCR_KW), "\n")   # same, with event times
print(pd.DataFrame(events).to_string() if events else "no events detected")
from IPython.display import Audio, display
display(Audio(AUDIO))
''')


# ----------------------------------------------------------------------------------------------- write
def build(variant, path):
    cells = []
    for v, kind, src in CELLS:
        if v not in ("both", variant): continue
        lines = src.split("\n")
        c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code": c.update(execution_count=None, outputs=[])
        cells.append(c)
    for i, c in enumerate(cells): c["id"] = f"{variant}-{i:02d}"
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if variant == "colab": meta["accelerator"] = "GPU"; meta["colab"] = {"gpuType": "T4", "provenance": []}
    json.dump({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5},
              open(path, "w"), indent=1, ensure_ascii=False)
    print("wrote", path, len(cells), "cells")


def to_script(variant, path, smoke_overrides=""):
    # For testing: concatenated code cells; notebook magics commented out.
    out = []
    for v, kind, src in CELLS:
        if kind != "code" or v not in ("both", variant): continue
        out.append("\n".join(("# " + l) if l.lstrip().startswith(("!", "%")) else l for l in src.split("\n")))
        out.append(f"print('=== cell done', flush=True)\n")
    open(path, "w").write("\n".join(out))


if __name__ == "__main__":
    build("local", f"{OUT_DIR}/train_local.ipynb")
    build("colab", f"{OUT_DIR}/train_colab.ipynb")
    if len(sys.argv) > 1:
        to_script(sys.argv[1], sys.argv[2])
