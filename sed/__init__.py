"""Speech event detection: non-verbal vocal sounds and background noise with timestamps, for rich transcripts."""
from .audio import load_audio
from .detector import Detector, class_runs
from .taxonomy import BACKGROUND_EVENTS, EVENTS, INLINE, SPEAKER_EVENTS
from .transcript import clean, rich_transcript

__version__ = "1.0.0"
__all__ = ["Detector", "class_runs", "load_audio", "clean", "rich_transcript",
           "EVENTS", "INLINE", "SPEAKER_EVENTS", "BACKGROUND_EVENTS", "__version__"]
