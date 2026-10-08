"""Audio loading: anything ffmpeg reads -> 16 kHz mono float32."""
import subprocess

import numpy as np

SR = 16000


def load_audio(path, sr=SR):
    # Same decoding as openai-whisper's load_audio, so detector and ASR see identical samples.
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found: install it (apt install ffmpeg / brew install ffmpeg)") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg could not decode {path}: {e.stderr.decode(errors='ignore')[-300:]}") from None
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
