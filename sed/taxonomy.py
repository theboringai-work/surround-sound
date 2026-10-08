"""Canonical event tags: dataset labels -> one tag per concept, and how tags are rendered."""
import re

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


# deepvk/NonverbalTTS writes each sound as an emoji inside the transcript (grunt is merged into groan).
TTS_EMOJI = {"🌬": "breath", "🤣": "laugh", "👃": "sniff", "😷": "cough", "🗣": "throat_clear",
             "😤": "sigh", "😖": "groan", "🤧": "sneeze", "😴": "snore", "🐖": "groan"}

# OpenSLR 99 (Deeply Nonverbal Vocalization) folder names; None = not used.
SLR_MAP = {"coughing": "cough", "yawning": "yawn", "throat-clearing": "throat_clear", "sighing": "sigh",
           "lip-smacking": "lip_smack", "lip-popping": "lip_smack", "panting": "breath", "crying": "cry",
           "laughing": "laugh", "sneezing": "sneeze", "nose-blowing": "nose_blow", "moaning": "groan",
           "screaming": "scream", "tongue-clicking": "click", "teeth-chattering": None, "teeth-grinding": None}

# The event list (same as in train_local.ipynb / train_colab.ipynb). A checkpoint carries its own class list;
# these lists only decide how each tag is rendered: speaker sounds inline as [tag], background as <tag> ... </tag>.
SPEAKER_EVENTS = ["breath", "sigh", "sniff", "laugh", "cough", "throat_clear", "cry", "gasp", "yawn", "snore",
                  "sneeze", "nose_blow", "lip_smack", "groan", "scream", "whisper", "shout", "chew", "burp", "hiccup",
                  "click"]
BACKGROUND_EVENTS = ["horn", "vehicle", "siren", "bird", "insect", "dog_bark", "animal", "child_voice", "baby_cry",
                     "music", "tv_speaker", "machine", "fan", "phone_ring", "phone_vibrate", "beep", "bell", "whistle",
                     "buzzer", "applause", "cheer", "crowd", "footsteps", "wind", "rain", "thunder", "water", "knock",
                     "door", "dishes", "glass_break", "gunshot", "explosion", "construction"]
EVENTS = SPEAKER_EVENTS + BACKGROUND_EVENTS
INLINE = set(SPEAKER_EVENTS)
