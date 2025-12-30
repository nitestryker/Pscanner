import os
import asyncio
import json
import queue
import time
from pathlib import Path
import re
import html as htmlmod
from urllib.parse import quote

import numpy as np
import sounddevice as sd
import websockets

# =============================================================================
# CONFIG
# =============================================================================

# Prefer using env var:
#   setx DEEPGRAM_API_KEY "yourkey"
#   then: DEEPGRAM_KEY = os.environ["DEEPGRAM_API_KEY"]
DEEPGRAM_KEY = "4ac21a8943641cc4151b3955ace862c7868d708e"

# -------------------------
# Deepgram / audio settings
# -------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
ENDPOINTING_MS = 1000

DG_BASE_URL = (
    "wss://api.deepgram.com/v1/listen"
    f"?model=nova-3"
    f"&language=en-US"
    f"&encoding=linear16"
    f"&sample_rate={SAMPLE_RATE}"
    f"&channels={CHANNELS}"
    f"&smart_format=true"
    f"&numerals=true"
    f"&interim_results=true"
    f"&endpointing={ENDPOINTING_MS}"
    f"&utterance_end_ms=1000"
)

# Tight list of keyterms (helps Deepgram bias without making it too noisy)
PHONETIC_UNITS = [
    "Adam", "Boy", "Charles", "David", "Edward", "Frank", "George", "Henry",
    "Ida", "John", "King", "Lincoln", "Mary", "Nora", "Ocean", "Paul", "Queen",
    "Robert", "Sam", "Tom", "Union", "Victor", "William", "X[- ]?Ray", "Yellow", "Zebra",
    "Echo",
]
KEYTERMS = [
    "Adam", "Boy", "Charles", "David", "Edward", "Frank", "George", "Henry", "Ida",
    "John", "King", "Lincoln", "Mary", "Nora", "Ocean", "Paul", "Queen", "Robert",
    "Sam", "Tom", "Union", "Victor", "William", "X-ray", "Xray", "Yellow", "Zebra", "Echo",
    "dispatch", "copy", "repeat", "standby", "code",
    "ten four", "ten-seven", "ten eight", "ten twenty eight",
    "eleven twenty five",
]

def build_dg_url() -> str:
    parts = [DG_BASE_URL]
    for k in KEYTERMS:
        parts.append("&keyterm=" + quote(k))
    return "".join(parts)

DG_URL = build_dg_url()

audio_q: "queue.Queue[bytes]" = queue.Queue()

# -------------------------
# Logging fallback when finals never arrive
# -------------------------
FORCE_LOG_AFTER_SECONDS = 6.0
FORCE_LOG_MIN_INTERVAL = 4.0

# =============================================================================
# CB / radio-style tuning (ASR-safe defaults)
# =============================================================================
TUNE_ENABLED = True
HP_HZ = 250.0
LP_HZ = 3600.0

GATE_ENABLED = True
GATE_RMS = 0.006
GATE_ATTENUATION = 0.35

AGC_ENABLED = True
AGC_TARGET_RMS = 0.045
AGC_MAX_GAIN = 8.0
AGC_MIN_GAIN = 0.25

LIMIT_ENABLED = False
LIMIT_THRESHOLD = 0.85

SOFTCLIP_ENABLED = False

PREEMPH_ENABLED = True
PREEMPH = 0.85

# =============================================================================
# PATHS (anchored to script directory)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
OBS_DIR = BASE_DIR / "obs_text"
OBS_DIR.mkdir(parents=True, exist_ok=True)

OBS_LIVE_FILE = OBS_DIR / "live_caption.txt"
OBS_FINAL_FILE = OBS_DIR / "final_caption.txt"
OBS_CAPTION_LOG_FILE = OBS_DIR / "caption_log.txt"
LIVE_MAX_CHARS = 300

FULL_LOG_FILE = OBS_DIR / "full_transcript_log.txt"
FULL_LOG_HTML_FILE = OBS_DIR / "full_transcript_log.html"

SILENCE_GAP_SECONDS = 4.0
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# LOWER THIRD MODE (HTML)
# =============================================================================
LOWER_THIRD_MODE = True

# OBS Browser Source size suggestion:
#   Width: 1920
#   Height: 360
LOWER_THIRD_WIDTH = 1920
LOWER_THIRD_HEIGHT = 360

# keep just the most recent blocks visible in lower-third
LOWER_THIRD_MAX_BLOCKS = 6

# =============================================================================
# 10 / 11 CODES (meaning annotations)
# =============================================================================
CODE_MEANINGS: dict[str, str] = {
    "904": "Fire (specify)",
    "911UNK": "Unknown 911 calls",
    "952": "Report on conditions",

    "10-1": "Receiving poorly",
    "10-2": "Receiving OK",
    "10-3": "Change channels",
    "10-4": "Message received and understood",
    "10-5": "Relay to",
    "10-6": "Busy, standby",
    "10-7": "Out of service",
    "10-7B": "Out of service, personal",
    "10-7CT": "Out of service, court",
    "10-7FU": "Out of service, follow up",
    "10-7OD": "Off duty",
    "10-7RW": "Out of service, report writing",
    "10-7T": "Out of service, training",
    "10-8": "In service",
    "10-8FU": "Follow up, but available",
    "10-9": "Repeat",
    "10-10": "Out of service, home",
    "10-12": "Visitor or official present",
    "10-14": "Escort",
    "10-15": "Prisoner in custody",
    "10-16": "Pick-up",
    "10-19": "En-route to station",
    "10-20": "Location",
    "10-21": "Phone",
    "10-22": "Cancel",
    "10-23": "Standby",
    "10-27": "Request driver license info",
    "10-28": "Request registration info",
    "10-29": "Check wanted vehicle or property",
    "10-29A": "Warrant check",
    "10-29C": "Complete check (warrants & history)",
    "10-32": "Drowning",
    "10-33A": "Audible alarm",
    "10-33S": "Silent alarm",
    "10-34": "Open door",
    "10-35": "Open window",
    "10-36": "Confidential Info",
    "10-45": "Injured person",
    "10-46": "Sick person",
    "10-49": "En route to event",
    "10-50": "Take a report",
    "10-51": "Intoxicated person",
    "10-53": "Person down",
    "10-54": "Possible dead body",
    "10-55": "Coroner’s case",
    "10-56": "Suicide",
    "10-56A": "Attempted suicide",
    "10-57": "Firearms discharge",
    "10-58": "Garbage complaint",
    "10-62": "Meet the citizen",
    "10-62FD": "Citizen flag-down",
    "10-65": "Missing person",
    "10-65F": "Found missing person",
    "10-65J": "Missing juvenile",
    "10-65JX": "Missing female juvenile",
    "10-65MH": "Missing person, mentally handicapped",
    "10-66": "Suspicious person",
    "10-66P": "Suspicious package",
    "10-66W": "Suspicious person with a weapon",
    "10-66X": "Suspicious female",
    "10-67": "Person calling for help",
    "10-70": "Prowler",
    "10-71": "Person shot",
    "10-72": "Person stabbed",
    "10-73": "How do you receive?",
    "10-80": "Explosion",
    "10-86": "Any traffic?",
    "10-87": "Meet the officer",
    "10-91": "Stray animal",
    "10-91A": "Vicious animal",
    "10-91B": "Noisy animal",
    "10-91C": "Injured animal",
    "10-91D": "Dead animal",
    "10-95": "Pedestrian stop",
    "10-96": "Pedestrian stop – High Risk",
    "10-97": "Arrived at assignment",
    "10-98": "Completed last assignment",

    "11-24": "Abandoned vehicle",
    "11-25": "Traffic hazard",
    "11-54": "Suspicious vehicle",
    "11-79": "Vehicle accident - Ambulance en route",
    "11-80": "Vehicle accident - Major injury",
    "11-81": "Vehicle accident - Minor injury",
    "11-82": "Vehicle accident - Property damage only",
    "11-83": "Vehicle accident - Unknown injury",
    "11-84": "Traffic control",
    "11-85": "Tow truck needed",
    "11-95": "Vehicle stop",
    "11-96": "Vehicle stop – High Risk",

    # ✅ Updated per your request
    "11-98": "Meet (another officer / RP / etc.)",
}

# =============================================================================
# PC CODES (subset)
# =============================================================================
PC_MEANINGS: dict[str, str] = {
    "835": "Method of Arrest",
    "835a": "Effecting Arrest; Resistance",
    "484": "Theft (DEFINED)",
    "487(a)": "Grand Theft (FELONY)",
    "488": "Petty Theft (MISDEMEANOR)",
    "211": "Robbery (FELONY)",
    "215": "Carjacking (FELONY)",
    "240": "Assault (MISDEMEANOR)",
    "242": "Battery (MISDEMEANOR)",
    "243(f)(4)": "Serious bodily injury defined (FELONY)",
    "245": "Assault with a deadly weapon (FELONY)",
    "459": "Burglary (FELONY)",
    "602": "Trespassing (MISDEMEANOR)",
}

# =============================================================================
# KEYWORD / HIGHLIGHT CONFIG
# =============================================================================
ALERT_KEYWORDS = [
    "Code 3", "Code 20", "Code 30", "Code 33",
    "Code 6A", "Code 6D", "Code 6F", "Code 6H", "Code 6M",
    "10-71", "10-72", "10-53", "10-54", "10-55", "10-56", "10-57",
    "10-80", "10-45",
]
LOCATIONS: list[str] = []

PATTERN_ALERTS = re.compile(
    r"\b(" + "|".join(map(re.escape, ALERT_KEYWORDS)) + r")\b",
    re.IGNORECASE
) if ALERT_KEYWORDS else None

PATTERN_LOCATIONS = re.compile(
    r"\b(" + "|".join(map(re.escape, LOCATIONS)) + r")\b",
    re.IGNORECASE
) if LOCATIONS else None

# =============================================================================
# CALLSIGN REGEX RULES
# =============================================================================
NUM_WORDS = r"(?:\d{1,4}|one|won|two|to|too|three|four|for|ford|forth|five|six|seven|eight|ate|nine|ten)"

CALLSIGN_SPACED = re.compile(
    r"\b(" + "|".join(PHONETIC_UNITS) + r")\s+(" + NUM_WORDS + r")\b",
    re.IGNORECASE
)
CALLSIGN_JOINED = re.compile(
    r"\b(" + "|".join(PHONETIC_UNITS) + r")(\d{2,4})\b",
    re.IGNORECASE
)
CALLSIGN_MULTI = re.compile(
    r"\b(\d{0,2})\s*(" + "|".join(PHONETIC_UNITS) + r")(?:\s*(" + "|".join(PHONETIC_UNITS) + r")){1,4}\s*(\d{1,4})\b",
    re.IGNORECASE
)

# =============================================================================
# 10/11 CODE DETECTION (numeric + spoken)  ✅ UPDATED to include joined forms
# =============================================================================
PATTERN_CODE_ANY_NUMERIC = re.compile(
    r"\b("
    r"(?:10|11)\s*[- ]\s*\d{1,3}\s*[A-Z]{0,3}"   # 10-23 / 10 23 / 11-98
    r"|(?:10|11)\d{1,3}[A-Z]{0,3}"               # 10023 / 11098
    r"|911UNK|904|952"
    r")\b",
    re.IGNORECASE
)

NUMBER_WORDS = [
    "zero", "oh",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred",
]
PATTERN_CODE_ANY_WORDS = re.compile(
    r"\b(ten|eleven)\s+((?:"
    + "|".join(NUMBER_WORDS)
    + r")(?:[-\s]+(?:"
    + "|".join(NUMBER_WORDS)
    + r")){0,3})\b",
    re.IGNORECASE
)

_WORD_TO_VAL = {
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

def _parse_number_words_phrase(phrase: str) -> int | None:
    tokens = re.split(r"[-\s]+", phrase.lower().strip())
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    current = 0
    seen = False
    for t in tokens:
        v = _WORD_TO_VAL.get(t)
        if v is None:
            return None
        seen = True
        if v == 100:
            if current == 0:
                current = 1
            current *= 100
        else:
            current += v
    return current if seen else None

def convert_spoken_codes_to_numeric(text: str) -> str:
    if not text:
        return text

    def repl(m: re.Match) -> str:
        prefix_word = m.group(1).lower()
        rest_phrase = m.group(2)
        prefix = "10" if prefix_word == "ten" else "11" if prefix_word == "eleven" else ""
        n = _parse_number_words_phrase(rest_phrase)
        if not prefix or n is None:
            return m.group(0)
        candidate = f"{prefix}-{n}"
        return candidate if candidate in CODE_MEANINGS else m.group(0)

    return PATTERN_CODE_ANY_WORDS.sub(repl, text)

# ✅ NEW: convert joined numeric codes like 11098 -> 11-98 (only if it’s a known code)
JOINED_10_11 = re.compile(r"\b(?P<prefix>10|11)(?P<body>\d{1,3})(?P<suffix>[A-Z]{0,3})\b", re.IGNORECASE)

def convert_joined_numeric_codes(text: str) -> str:
    if not text:
        return text

    def repl(m: re.Match) -> str:
        prefix = m.group("prefix")
        body_raw = m.group("body")
        suffix = (m.group("suffix") or "").upper()

        try:
            body_int = int(body_raw)
        except ValueError:
            return m.group(0)

        candidate = f"{prefix}-{body_int}{suffix}"
        if candidate in CODE_MEANINGS:
            return candidate

        return m.group(0)

    return JOINED_10_11.sub(repl, text)

def normalize_code_key(raw_code: str) -> str:
    s = raw_code.strip()
    up = s.upper()
    if up in ("904", "952", "911UNK"):
        return up
    up = re.sub(r"\s+", " ", up)
    up = up.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    up = up.replace(" ", "-")
    up = re.sub(r"\b(10|11)-(\d{1,3})-([A-Z]{1,3})\b", r"\1-\2\3", up)
    return up

# ✅ UPDATED: joined -> spoken -> annotate
def annotate_codes(text: str) -> str:
    if not text:
        return text

    text = convert_joined_numeric_codes(text)
    text = convert_spoken_codes_to_numeric(text)

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        key = normalize_code_key(raw)
        meaning = CODE_MEANINGS.get(key)
        if not meaning:
            return raw
        return f"{raw} ({meaning})"

    return PATTERN_CODE_ANY_NUMERIC.sub(repl, text)

# =============================================================================
# "I'm 97" shorthand => treat as 10-97 if it exists
# =============================================================================
IM_SHORTHAND_NUMERIC = re.compile(
    r"\b(?P<prefix>I'?m|I am|we'?re|we are)\s+(?P<num>\d{1,3})\b",
    re.IGNORECASE
)
IM_SHORTHAND_WORDS = re.compile(
    r"\b(?P<prefix>I'?m|I am|we'?re|we are)\s+(?P<words>(?:"
    + "|".join(NUMBER_WORDS)
    + r")(?:[-\s]+(?:"
    + "|".join(NUMBER_WORDS)
    + r")){0,3})\b",
    re.IGNORECASE
)

def annotate_im_shorthand(text: str) -> str:
    if not text:
        return text

    def repl_num(m: re.Match) -> str:
        prefix = m.group("prefix")
        num = int(m.group("num"))
        key = f"10-{num}"
        meaning = CODE_MEANINGS.get(key)
        if not meaning:
            return m.group(0)
        return f"{prefix} {num} ({meaning})"

    text = IM_SHORTHAND_NUMERIC.sub(repl_num, text)

    def repl_words(m: re.Match) -> str:
        prefix = m.group("prefix")
        words = m.group("words")
        n = _parse_number_words_phrase(words)
        if n is None:
            return m.group(0)
        key = f"10-{n}"
        meaning = CODE_MEANINGS.get(key)
        if not meaning:
            return m.group(0)
        return f"{prefix} {n} ({meaning})"

    text = IM_SHORTHAND_WORDS.sub(repl_words, text)
    return text

# =============================================================================
# PC CODE DETECTION + ANNOTATION
# =============================================================================
PC_REF_1 = re.compile(r"\b(?P<code>\d{1,4}(?:\.\d+)?[a-z]?(?:\([^)]+\))?)\s*PC\b", re.IGNORECASE)
PC_REF_2 = re.compile(r"\bPC\s*(?P<code>\d{1,4}(?:\.\d+)?[a-z]?(?:\([^)]+\))?)\b", re.IGNORECASE)

def annotate_pc_codes(text: str) -> str:
    if not text:
        return text

    def repl1(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = PC_MEANINGS.get(raw_code) or PC_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"{raw_code} PC ({meaning})"

    def repl2(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = PC_MEANINGS.get(raw_code) or PC_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"PC {raw_code} ({meaning})"

    text = PC_REF_1.sub(repl1, text)
    text = PC_REF_2.sub(repl2, text)
    return text

# =============================================================================
# 10-28 "INFO LOOKUP" (phonetic decode)
# =============================================================================
LOOKUP_TRIGGER = re.compile(
    r"\b(10[\s-]?28|1028|ten\s+twenty\s+eight|ten\s+28)\b",
    re.IGNORECASE
)

LOOKUP_STOP = re.compile(
    r"\b("
    r"dob|d\.o\.b|date\s+of\s+birth|birth\s+date|"
    r"cii|c\.i\.i|"
    r"returns?|returned|returning|comes?\s+back|"
    r"record|ro|wants?|probation|parole|"
    r"no\s+record|negative|clear|confirmed"
    r")\b",
    re.IGNORECASE
)

LOOKUP_SEPARATORS = re.compile(
    r"\b(space|last\s+name|surname|family\s+name|first\s+name|middle\s+name|last)\b",
    re.IGNORECASE
)

PHONETIC_TO_LETTER = {
    "adam": "A", "boy": "B", "charles": "C", "david": "D", "edward": "E",
    "frank": "F", "george": "G", "henry": "H", "ida": "I", "john": "J",
    "king": "K", "lincoln": "L", "mary": "M", "nora": "N", "ocean": "O",
    "paul": "P", "queen": "Q", "robert": "R", "sam": "S", "tom": "T",
    "union": "U", "victor": "V", "william": "W",
    "xray": "X", "x-ray": "X", "x": "X",
    "yellow": "Y", "zebra": "Z",
    "echo": "E",
}

def normalize_phonetic_token(tok: str) -> str:
    t = tok.lower().strip()
    t = t.replace(".", "").replace(",", "").replace(";", "").replace(":", "")
    t = t.replace("x ray", "xray").replace("x-ray", "xray")
    return t

def is_number_token(tok: str) -> bool:
    t = tok.lower().strip()
    return re.fullmatch(r"(?:\d{1,4}|one|won|two|to|too|three|four|for|ford|forth|five|six|seven|eight|ate|nine|ten)", t, re.IGNORECASE) is not None

class InfoLookupDecoder:
    def __init__(self, window_seconds: float = 14.0, min_letters_per_word: int = 3):
        self.window_seconds = window_seconds
        self.min_letters_per_word = min_letters_per_word
        self.active_until: float = 0.0
        self.words: list[str] = []
        self.current_letters: list[str] = []

    def reset(self):
        self.active_until = 0.0
        self.words = []
        self.current_letters = []

    def _finalize_current_word(self):
        if self.current_letters:
            w = "".join(self.current_letters)
            self.current_letters = []
            if len(w) >= self.min_letters_per_word:
                self.words.append(w)

    def _extract_letters(self, transcript: str) -> list[str]:
        raw = re.split(r"\s+", transcript.strip())
        out: list[str] = []
        i = 0
        while i < len(raw):
            tok = normalize_phonetic_token(raw[i])
            if tok in PHONETIC_TO_LETTER:
                # skip callsigns like "Adam 12"
                if i + 1 < len(raw):
                    nxt = normalize_phonetic_token(raw[i + 1])
                    if is_number_token(nxt):
                        i += 2
                        continue
                out.append(PHONETIC_TO_LETTER[tok])
            i += 1
        return out

    def _emit_if_ready(self) -> str | None:
        self._finalize_current_word()
        if len(self.words) >= 2:
            result = " ".join(self.words[:2])
            self.reset()
            return result
        return None

    def process_final(self, transcript: str, now: float) -> str | None:
        if not transcript:
            return None

        if LOOKUP_TRIGGER.search(transcript):
            self.active_until = max(self.active_until, now + self.window_seconds)

        if now > self.active_until:
            self.reset()
            return None

        stop_hit = LOOKUP_STOP.search(transcript) is not None
        sep_hit = LOOKUP_SEPARATORS.search(transcript) is not None

        letters = self._extract_letters(transcript)
        if letters:
            self.current_letters.extend(letters)

        if sep_hit:
            self._finalize_current_word()

        if stop_hit:
            out = self._emit_if_ready()
            self.reset()
            return out

        out = self._emit_if_ready()
        if out:
            return out

        return None

lookup_decoder = InfoLookupDecoder()

# =============================================================================
# FILE IO (Windows / OBS lock-safe)
# =============================================================================
def atomic_write(path: Path, text: str, retries: int = 20, delay: float = 0.03) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    for _ in range(retries):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
        except OSError:
            time.sleep(delay)
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass

def append_flush_fsync(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

def contains_alert(text: str) -> bool:
    if not text or PATTERN_ALERTS is None:
        return False
    return PATTERN_ALERTS.search(text) is not None

def _maybe_split_joined_digits(digits: str):
    if len(digits) == 4 and digits.isdigit():
        unit_digit = digits[0]
        tail = digits[1:]
        tail_i = int(tail)
        if 100 <= tail_i <= 199:
            return unit_digit, tail
    return None, None

# =============================================================================
# HTML HIGHLIGHTING
# =============================================================================
def highlight_to_html(text: str) -> str:
    if not text:
        return ""

    safe = htmlmod.escape(text)

    if PATTERN_ALERTS:
        safe = PATTERN_ALERTS.sub(r"<span class='hl alert'>\1</span>", safe)

    safe = CALLSIGN_SPACED.sub(r"<span class='hl unit'>\g<0></span>", safe)

    def repl_joined(m: re.Match) -> str:
        unit = m.group(1)
        digits = m.group(2)
        unit_digit, tail = _maybe_split_joined_digits(digits)
        if unit_digit and tail:
            return (
                f"<span class='hl unit'>{htmlmod.escape(unit)} {htmlmod.escape(unit_digit)}</span> "
                f"<span class='hl ten'>{htmlmod.escape(tail)}</span>"
            )
        return f"<span class='hl unit'>{htmlmod.escape(m.group(0))}</span>"

    safe = CALLSIGN_JOINED.sub(repl_joined, safe)
    safe = CALLSIGN_MULTI.sub(r"<span class='hl unit'>\g<0></span>", safe)

    safe = PATTERN_CODE_ANY_NUMERIC.sub(r"<span class='hl ten'>\g<0></span>", safe)
    safe = LOOKUP_TRIGGER.sub(r"<span class='hl lookup'>\1</span>", safe)

    safe = re.sub(
        r"\bPC\s*\d{1,4}(?:\.\d+)?[a-z]?(?:\([^)]+\))?\b",
        lambda m: f"<span class='hl ten'>{m.group(0)}</span>",
        safe,
        flags=re.IGNORECASE,
    )

    if PATTERN_LOCATIONS:
        safe = PATTERN_LOCATIONS.sub(r"<span class='hl loc'>\1</span>", safe)

    return safe

# =============================================================================
# OBS + LOG WRITERS
# =============================================================================
class OBSCaptionWriter:
    def __init__(self):
        self.last_live = ""

    def update_live(self, text: str):
        text = text.strip()
        if not text:
            return
        if len(text) > LIVE_MAX_CHARS:
            text = text[-LIVE_MAX_CHARS:]
        if text == self.last_live:
            return
        self.last_live = text
        atomic_write(OBS_LIVE_FILE, text)

    def write_final(self, text: str):
        text = text.strip()
        if not text:
            return
        atomic_write(OBS_FINAL_FILE, text)
        ts = time.strftime(TS_FORMAT)
        append_flush_fsync(OBS_CAPTION_LOG_FILE, f"[{ts}] {text}\n")

class FullTranscriptLogger:
    def __init__(self, txt_path: Path, html_path: Path, gap_seconds: float):
        self.txt_path = txt_path
        self.html_path = html_path
        self.gap_seconds = gap_seconds
        self.last_write_time: float | None = None
        self.blocks: list[dict] = []
        self.max_blocks = 300
        self._write_html()

    def _ts(self) -> str:
        return time.strftime(TS_FORMAT)

    def add_entry(self, text: str, kind: str = "final", lookup_decoded: str | None = None):
        text = text.strip()
        if not text:
            return

        now = time.time()
        start_new_entry = (
            self.last_write_time is None
            or (now - self.last_write_time) >= self.gap_seconds
        )

        if start_new_entry:
            append_flush_fsync(self.txt_path, f"[{self._ts()}] {text}\n")
        else:
            append_flush_fsync(self.txt_path, f"    {text}\n")

        if lookup_decoded:
            append_flush_fsync(self.txt_path, f"    INFO LOOKUP: {lookup_decoded}\n")

        if start_new_entry or not self.blocks:
            self.blocks.append({"ts": self._ts(), "lines": [], "lookups": []})

        line_to_store = text + (" [partial]" if kind == "partial" else "")
        self.blocks[-1]["lines"].append(line_to_store)

        if lookup_decoded:
            self.blocks[-1]["lookups"].append(lookup_decoded)

        if len(self.blocks) > self.max_blocks:
            self.blocks = self.blocks[-self.max_blocks:]

        self.last_write_time = now
        self._write_html()

    def _write_html(self):
        blocks = self.blocks
        if LOWER_THIRD_MODE and len(blocks) > LOWER_THIRD_MAX_BLOCKS:
            blocks = blocks[-LOWER_THIRD_MAX_BLOCKS:]

        parts = []
        parts.append("<!doctype html>")
        parts.append("<html><head><meta charset='utf-8'>")
        parts.append("""
<style>
  :root {
    --bg: rgba(0,0,0,0.55);
    --bubble: rgba(255,255,255,0.08);
    --text: #ffffff;
  }

  html, body {
    margin:0; padding:0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
    font-family: Arial, sans-serif;
    color: var(--text);
  }

  .stage {
    position: relative;
    width: 100%;
    height: 100%;
    padding: 18px 24px;
    box-sizing: border-box;
  }

  .stack {
    position: absolute;
    left: 24px;
    right: 24px;
    bottom: 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .block {
    padding: 12px 14px;
    border-radius: 14px;
    background: var(--bg);
    box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    backdrop-filter: blur(6px);
  }

  .ts {
    font-size: 18px;
    opacity: 0.80;
    margin-bottom: 6px;
    font-weight: 700;
    letter-spacing: 0.2px;
  }

  .line {
    font-size: 28px;
    line-height: 1.20;
    margin: 0 0 6px 0;
    font-weight: 700;
    text-shadow: 0 2px 10px rgba(0,0,0,0.55);
  }

  .lookupline {
    margin-top: 8px;
    padding: 10px 12px;
    border-radius: 12px;
    background: var(--bubble);
    font-size: 24px;
    font-weight: 700;
    text-shadow: 0 2px 10px rgba(0,0,0,0.55);
  }

  .hl { padding: 0 8px; border-radius: 10px; font-weight: 900; }
  .hl.alert  { background: rgba(255, 0, 0, 0.35); }
  .hl.unit   { background: rgba(255, 255, 0, 0.25); }
  .hl.ten    { background: rgba(0, 200, 255, 0.22); }
  .hl.loc    { background: rgba(200, 150, 255, 0.20); }
  .hl.lookup { background: rgba(0, 255, 150, 0.18); }

  .block[data-age="old"] { opacity: 0.82; }
</style>
</head><body>
""".strip())

        parts.append("<div class='stage'>")
        parts.append("<div class='stack' id='stack'>")

        for idx, b in enumerate(blocks):
            age = "old" if idx < len(blocks) - 1 else "new"
            parts.append(f"<div class='block' data-age='{age}'>")
            parts.append(f"<div class='ts'>{htmlmod.escape(b['ts'])}</div>")

            for line in b["lines"]:
                parts.append(f"<div class='line'>{highlight_to_html(line)}</div>")

            for decoded in b.get("lookups", []):
                parts.append(
                    f"<div class='lookupline'><span class='hl lookup'>INFO LOOKUP:</span> {htmlmod.escape(decoded)}</div>"
                )

            parts.append("</div>")

        parts.append("</div></div>")

        parts.append("""
<script>
(function(){
  function pinBottom(){
    try {
      window.scrollTo(0, document.body.scrollHeight);
    } catch(e){}
  }
  pinBottom();
  setInterval(pinBottom, 250);
  setInterval(function(){
    location.reload();
  }, 1500);
})();
</script>
""".strip())

        parts.append("</body></html>")
        atomic_write(self.html_path, "\n".join(parts))

obs_writer = OBSCaptionWriter()
full_logger = FullTranscriptLogger(FULL_LOG_FILE, FULL_LOG_HTML_FILE, SILENCE_GAP_SECONDS)

# =============================================================================
# RADIO TUNER DSP
# =============================================================================
class RadioTuner:
    def __init__(self, sr: int):
        self.sr = sr
        self.hp_y = 0.0
        self.hp_x_prev = 0.0
        self.lp_y = 0.0
        self.pre_x_prev = 0.0
        self._update_coeffs()

    def _update_coeffs(self):
        dt = 1.0 / self.sr
        rc_hp = 1.0 / (2.0 * np.pi * max(1.0, HP_HZ))
        self.hp_a = rc_hp / (rc_hp + dt)
        rc_lp = 1.0 / (2.0 * np.pi * max(1.0, LP_HZ))
        self.lp_b = dt / (rc_lp + dt)

    def pre_emphasis(self, x: np.ndarray) -> np.ndarray:
        if not PREEMPH_ENABLED:
            return x
        y = np.empty_like(x)
        prev = self.pre_x_prev
        a = PREEMPH
        for i in range(len(x)):
            xi = x[i]
            y[i] = xi - a * prev
            prev = xi
        self.pre_x_prev = float(prev)
        return y

    def high_pass(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        a = self.hp_a
        y_prev = self.hp_y
        x_prev = self.hp_x_prev
        for i in range(len(x)):
            xi = x[i]
            yi = a * (y_prev + xi - x_prev)
            y[i] = yi
            y_prev = yi
            x_prev = xi
        self.hp_y = float(y_prev)
        self.hp_x_prev = float(x_prev)
        return y

    def low_pass(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        b = self.lp_b
        y_prev = self.lp_y
        for i in range(len(x)):
            xi = x[i]
            y_prev = y_prev + b * (xi - y_prev)
            y[i] = y_prev
        self.lp_y = float(y_prev)
        return y

    def noise_gate(self, x: np.ndarray) -> np.ndarray:
        if not GATE_ENABLED:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms < GATE_RMS:
            return x * GATE_ATTENUATION
        return x

    def agc(self, x: np.ndarray) -> np.ndarray:
        if not AGC_ENABLED:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms <= 1e-6:
            return x
        gain = AGC_TARGET_RMS / rms
        gain = float(np.clip(gain, AGC_MIN_GAIN, AGC_MAX_GAIN))
        return x * gain

    def limiter(self, x: np.ndarray) -> np.ndarray:
        if not LIMIT_ENABLED:
            return x
        return np.clip(x, -LIMIT_THRESHOLD, LIMIT_THRESHOLD)

    def softclip(self, x: np.ndarray) -> np.ndarray:
        if not SOFTCLIP_ENABLED:
            return x
        return np.tanh(2.2 * x) / np.tanh(2.2)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not TUNE_ENABLED:
            return x
        x = self.pre_emphasis(x)
        x = self.high_pass(x)
        x = self.low_pass(x)
        x = self.noise_gate(x)
        x = self.agc(x)
        x = self.limiter(x)
        x = self.softclip(x)
        return np.clip(x, -1.0, 1.0)

tuner = RadioTuner(SAMPLE_RATE)

def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    x = indata[:, 0].astype(np.float32)
    x = tuner.process(x)
    pcm16 = (x * 32767.0).astype(np.int16).tobytes()
    audio_q.put(pcm16)

# =============================================================================
# POST-PROCESS PIPELINE
# =============================================================================
def post_process_transcript(text: str) -> str:
    if not text:
        return text
    text = annotate_im_shorthand(text)
    text = annotate_codes(text)
    text = annotate_pc_codes(text)
    return text

# =============================================================================
# DEEPGRAM MESSAGE PARSING (fix channel dict/list issue)
# =============================================================================
def extract_transcript_and_final(data: dict) -> tuple[str, bool]:
    is_final = bool(data.get("is_final", False) or data.get("speech_final", False))

    channel = data.get("channel")
    alts = None

    if isinstance(channel, dict):
        alts = channel.get("alternatives")
    elif isinstance(channel, list) and channel:
        first = channel[0]
        if isinstance(first, dict):
            alts = first.get("alternatives")

    if not alts:
        alts = data.get("alternatives")

    if not isinstance(alts, list) or not alts:
        return "", is_final

    transcript = (alts[0].get("transcript") or "").strip()
    return transcript, is_final

# =============================================================================
# WEBSOCKET TASKS
# =============================================================================
async def sender(ws):
    while True:
        chunk = await asyncio.to_thread(audio_q.get)
        await ws.send(chunk)

async def receiver(ws):
    last_interim_best: str = ""
    last_interim_update_time = 0.0
    last_forced_log_time = 0.0

    async for msg in ws:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue

        transcript_raw, is_final = extract_transcript_and_final(data)
        if not transcript_raw:
            continue

        decoded_lookup = None
        if is_final:
            decoded_lookup = lookup_decoder.process_final(transcript_raw, time.time())

        transcript = post_process_transcript(transcript_raw)

        if is_final:
            print(transcript)
        else:
            print(transcript, end="\r", flush=True)

        alert = contains_alert(transcript_raw)
        caption_text = f"🚨 {transcript}" if alert else transcript

        if is_final:
            obs_writer.write_final(caption_text)
            obs_writer.update_live(caption_text)
        else:
            obs_writer.update_live(caption_text)

        now = time.time()

        if is_final:
            full_logger.add_entry(transcript, kind="final", lookup_decoded=decoded_lookup)
            last_interim_best = ""
            last_interim_update_time = 0.0
            continue

        if len(transcript) >= 8 and transcript != last_interim_best:
            last_interim_best = transcript
            last_interim_update_time = now

        if last_interim_best and last_interim_update_time:
            if (now - last_interim_update_time) >= FORCE_LOG_AFTER_SECONDS:
                if (now - last_forced_log_time) >= FORCE_LOG_MIN_INTERVAL:
                    full_logger.add_entry(last_interim_best, kind="partial", lookup_decoded=None)
                    last_forced_log_time = now
                    last_interim_best = ""
                    last_interim_update_time = 0.0

async def main():
    headers = {"Authorization": f"Token {DEEPGRAM_KEY}"}

    atomic_write(OBS_LIVE_FILE, "")
    atomic_write(OBS_FINAL_FILE, "")

    if not FULL_LOG_FILE.exists():
        FULL_LOG_FILE.write_text("", encoding="utf-8")

    if not FULL_LOG_HTML_FILE.exists():
        atomic_write(FULL_LOG_HTML_FILE, "<!doctype html><html><body></body></html>")

    blocksize = 320

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=blocksize,
    ):
        async with websockets.connect(DG_URL, additional_headers=headers) as ws:
            await asyncio.gather(sender(ws), receiver(ws))

if __name__ == "__main__":
    asyncio.run(main())
