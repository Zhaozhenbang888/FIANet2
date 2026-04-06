from dataclasses import dataclass
import re
from typing import Any, List, Tuple


@dataclass(frozen=True)
class PromptSpec:
    prompt: str
    target_phrases: Tuple[str, ...]
    position_phrases: Tuple[str, ...]


_CATEGORY_ALIASES = {
    "photovolatic": ("solar panel", "photovoltaic panel"),
    "ground track field": ("stadium",),
    "football court": ("soccer field",),
    "parking": ("parking lot",),
    "airplaneroad": ("runway",),
    "ocean": ("sea",),
    "pylon": ("tower",),
    "construction tower": ("tower crane",),
}


_ENGLISH_TARGET_ALIASES = {
    "car": ("car", "cars"),
    "ship": ("ship", "ships", "wheelbarrow", "wheelbarrows"),
    "photovolatic": (
        "photovolatic",
        "photovoltaic panel",
        "photovoltaic panels",
        "solar panel",
        "solar panels",
        "panel",
        "panels",
    ),
    "building": ("building", "buildings"),
    "airplane": ("airplane", "airplanes", "plane", "planes", "aircraft"),
    "road": ("road", "roads", "street", "streets"),
    "bridge": ("bridge", "bridges"),
    "basketball court": ("basketball court", "basketball courts"),
    "container": ("container", "containers", "shipping container", "shipping containers"),
    "train": ("train", "trains"),
    "pool": ("pool", "pools", "swimming pool", "swimming pools"),
    "storage tank": ("storage tank", "storage tanks", "tank", "tanks"),
    "tennis court": ("tennis court", "tennis courts"),
    "ground track field": (
        "ground track field",
        "ground track fields",
        "athletic field",
        "athletic fields",
        "running track",
        "running tracks",
        "red track",
        "gray track",
    ),
    "football court": ("football court", "football courts", "soccer field", "soccer fields"),
    "pylon": ("pylon", "pylons", "power line pylon", "power line pylons", "electricity pylon", "electricity pylons"),
    "digger": ("digger", "diggers", "excavator", "excavators"),
    "wind turbine": ("wind turbine", "wind turbines"),
    "construction tower": (
        "construction tower",
        "construction towers",
        "tower crane",
        "tower cranes",
        "site pylon",
        "site pylons",
        "building pylon",
        "building pylons",
    ),
    "road intersection": ("road intersection", "road intersections", "intersection", "intersections"),
    "parking": ("parking", "parking lot", "parking lots"),
    "baseball court": ("baseball court", "baseball courts", "baseball field", "baseball fields"),
    "river": ("river", "rivers"),
    "chimney": ("chimney", "chimneys"),
    "airplaneroad": ("airplaneroad", "runway", "runways"),
    "ocean": ("ocean", "sea"),
    "land": ("land",),
    "rugby court": ("rugby court", "rugby courts", "rugby field", "rugby fields"),
    "lake": ("lake", "lakes"),
    "grass": ("grass",),
    "badminton court": ("badminton court", "badminton courts"),
    "dam": ("dam", "dams"),
}


_SENTENCE_TEXT_KEYS = (
    "raw",
    "sent",
    "sentence",
    "text",
    "raw_sent",
    "sent_ch",
    "raw_ch",
)


TEXT_LANG_ENGLISH_ID = 0
TEXT_LANG_CHINESE_ID = 1


_DESCRIPTOR_PATTERNS = [
    ("\u6240\u6709", "all"),
    ("\u6700\u5927", "largest"),
    ("\u6700\u5c0f", "smallest"),
    ("\u8f83\u5927", "larger"),
    ("\u8f83\u5c0f", "smaller"),
    ("\u767d\u8272", "white"),
    ("\u9ed1\u8272", "black"),
    ("\u7ea2\u8272", "red"),
    ("\u84dd\u8272", "blue"),
    ("\u9ec4\u8272", "yellow"),
    ("\u7eff\u8272", "green"),
    ("\u7070\u8272", "gray"),
    ("\u6df1\u8272", "dark"),
    ("\u6d45\u8272", "light"),
    ("\u9752\u8272", "cyan"),
    ("\u7d2b\u8272", "purple"),
    ("\u68d5\u8272", "brown"),
    ("\u6a59\u7ea2\u8272", "orange red"),
    ("\u77e9\u5f62", "rectangular"),
    ("\u7403\u5f62", "circular"),
    ("\u5927\u8d27\u8f66", "large truck"),
    ("\u5927\u6c7d\u8f66", "large car"),
    ("\u516c\u4ea4\u8f66", "bus"),
]


_POSITION_PATTERNS = [
    ("\u5de6\u4e0a\u89d2", "top left"),
    ("\u53f3\u4e0a\u89d2", "top right"),
    ("\u5de6\u4e0b\u89d2", "bottom left"),
    ("\u53f3\u4e0b\u89d2", "bottom right"),
    ("\u5de6\u4fa7", "left"),
    ("\u53f3\u4fa7", "right"),
    ("\u5de6\u65b9", "left"),
    ("\u53f3\u65b9", "right"),
    ("\u9760\u5de6", "left"),
    ("\u9760\u53f3", "right"),
    ("\u9760\u4e0a", "top"),
    ("\u9760\u4e0b", "bottom"),
    ("\u6700\u4e0a\u8fb9", "topmost"),
    ("\u6700\u4e0b\u8fb9", "bottommost"),
    ("\u6700\u5de6\u4fa7", "leftmost"),
    ("\u6700\u53f3\u4fa7", "rightmost"),
    ("\u6700\u4e0a", "topmost"),
    ("\u6700\u4e0b", "bottommost"),
    ("\u6700\u5de6", "leftmost"),
    ("\u6700\u53f3", "rightmost"),
    ("\u4e0a\u8fb9", "top"),
    ("\u4e0b\u8fb9", "bottom"),
    ("\u4e0a\u4fa7", "top"),
    ("\u4e0b\u4fa7", "bottom"),
    ("\u5de6\u8fb9", "left"),
    ("\u53f3\u8fb9", "right"),
    ("\u4e0a\u9762", "top"),
    ("\u4e0a\u65b9", "top"),
    ("\u4e0b\u9762", "bottom"),
    ("\u4e0b\u65b9", "bottom"),
    ("\u4e2d\u95f4", "center"),
    ("\u9053\u8def\u4e0a", "on road"),
    ("\u9053\u8def\u4e0a\u7684", "on road"),
    ("\u6cb3\u6d41\u4e0a", "on river"),
    ("\u6d77\u9762\u4e0a", "on sea"),
    ("\u6d77\u4e0a", "on sea"),
    ("\u505c\u8f66\u573a\u5185", "in parking"),
    ("\u8dd1\u9053\u4e0a", "on runway"),
    ("\u8dd1\u9053\u4e0a\u7684", "on runway"),
    ("\u4ece\u4e0a\u5230\u4e0b", "top to bottom"),
    ("\u5c4b\u9876", "on roof"),
    ("\u9644\u8fd1", "nearby"),
]


_RELATION_PATTERNS = [
    ("\u6e38\u6cf3\u6c60", "with pool"),
    ("\u5149\u4f0f\u53d1\u7535\u677f", "with photovolatic"),
    ("\u96c6\u88c5\u7bb1", "with container"),
]


_ORDINAL_PATTERNS = [
    ("\u7b2c\u4e00", "first"),
    ("\u7b2c\u4e8c", "second"),
    ("\u7b2c\u4e09", "third"),
    ("\u7b2c\u56db", "fourth"),
    ("\u7b2c\u4e94", "fifth"),
    ("\u7b2c\u516d", "sixth"),
    ("\u7b2c\u4e03", "seventh"),
    ("\u7b2c\u516b", "eighth"),
    ("\u7b2c\u4e5d", "ninth"),
    ("\u7b2c\u5341", "tenth"),
]


_ENGLISH_POSITION_PATTERNS = [
    ("left side", "left"),
    ("on the left", "left"),
    ("to the left", "left"),
    ("left", "left"),
    ("right side", "right"),
    ("on the right", "right"),
    ("to the right", "right"),
    ("right", "right"),
    ("upper left", "top left"),
    ("upper right", "top right"),
    ("lower left", "bottom left"),
    ("lower right", "bottom right"),
    ("upper", "top"),
    ("top", "top"),
    ("below", "bottom"),
    ("bottom", "bottom"),
    ("center", "center"),
    ("middle", "center"),
    ("on the road", "on road"),
    ("on road", "on road"),
    ("on the river", "on river"),
    ("on river", "on river"),
    ("on the sea", "on sea"),
    ("on sea", "on sea"),
    ("in parking", "in parking"),
    ("in the parking lot", "in parking"),
    ("on the runway", "on runway"),
    ("on runway", "on runway"),
    ("on the roof", "on roof"),
    ("on roof", "on roof"),
    ("nearby", "nearby"),
]


def canonicalize_category_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def contains_ascii_letter(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def classify_text_language(text: str) -> str:
    has_cjk = contains_cjk(text)
    has_ascii = contains_ascii_letter(text)
    if has_cjk and has_ascii:
        return "other"
    if has_cjk:
        return "chinese"
    if has_ascii:
        return "english"
    return "other"


def strip_attached_ascii_noise(text: str) -> str:
    """Remove short ASCII fragments accidentally attached to CJK text.

    Examples:
      - "道路上的人行天桥s" -> "道路上的人行天桥"
      - "a道路" -> "道路"
    """
    if not text:
        return ""
    if not (contains_cjk(text) and contains_ascii_letter(text)):
        return text

    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])[A-Za-z]{1,3}$", "", text)
    cleaned = re.sub(r"^[A-Za-z]{1,3}(?=[\u4e00-\u9fff])", "", cleaned)
    return " ".join(cleaned.strip().split())


def sentence_to_text_language_id(text: str) -> int:
    language = classify_text_language(text)
    if language == "chinese":
        return TEXT_LANG_CHINESE_ID
    return TEXT_LANG_ENGLISH_ID


def text_matches_language_filter(text: str, language_filter: str) -> bool:
    if language_filter == "all":
        return True
    return classify_text_language(text) == language_filter


def _decode_bytes_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "gb18030", "latin1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _normalize_sentence_candidate(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        text = _decode_bytes_text(value)
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        tokens: List[str] = []
        for item in value:
            token = _normalize_sentence_candidate(item)
            if token:
                tokens.append(token)
        if not tokens:
            return ""
        if any(contains_cjk(token) for token in tokens):
            text = "".join(tokens)
        else:
            text = " ".join(tokens)
    else:
        text = str(value)

    return " ".join(text.strip().split())


def extract_sentence_text(sentence_entry: Any) -> str:
    if isinstance(sentence_entry, dict):
        for key in _SENTENCE_TEXT_KEYS:
            text = _normalize_sentence_candidate(sentence_entry.get(key))
            if text:
                return text
        return _normalize_sentence_candidate(sentence_entry.get("tokens"))
    return _normalize_sentence_candidate(sentence_entry)


def _append_unique(items: List[str], value: str) -> None:
    value = " ".join(value.split())
    if value and value not in items:
        items.append(value)


def _category_prompt(category_name: str) -> List[str]:
    category = canonicalize_category_name(category_name)
    parts = [category]
    for alias in _CATEGORY_ALIASES.get(category, ()):
        _append_unique(parts, alias)
    return parts


def _get_target_phrases(category_name: str, raw_text: str) -> Tuple[str, ...]:
    category = canonicalize_category_name(category_name)
    if contains_cjk(raw_text) and not contains_ascii_letter(raw_text):
        return (category,)

    phrases: List[str] = []
    for phrase in _ENGLISH_TARGET_ALIASES.get(category, (category,)):
        _append_unique(phrases, phrase)
    if not phrases:
        phrases.append(category)
    return tuple(phrases)


def _extract_descriptors(raw_text: str) -> List[str]:
    descriptors: List[str] = []
    for needle, english in _DESCRIPTOR_PATTERNS:
        if needle in raw_text:
            _append_unique(descriptors, english)
    if "\u884c\u9a76" in raw_text:
        _append_unique(descriptors, "moving")
    if "\u5411\u5de6\u884c\u9a76" in raw_text:
        _append_unique(descriptors, "moving left")
    if "\u5411\u53f3\u884c\u9a76" in raw_text:
        _append_unique(descriptors, "moving right")
    if "\u5411\u4e0a\u884c\u9a76" in raw_text:
        _append_unique(descriptors, "moving up")
    if "\u5411\u4e0b\u884c\u9a76" in raw_text:
        _append_unique(descriptors, "moving down")
    return descriptors


def _extract_positions(raw_text: str) -> List[str]:
    positions: List[str] = []
    has_cjk = contains_cjk(raw_text)
    has_ascii = contains_ascii_letter(raw_text)

    if has_ascii:
        for needle, english in _ENGLISH_POSITION_PATTERNS:
            if needle in raw_text:
                _append_unique(positions, english)
        if " row " in f" {raw_text} ":
            _append_unique(positions, "row")
        if " column " in f" {raw_text} ":
            _append_unique(positions, "column")

    if has_cjk:
        for needle, english in _POSITION_PATTERNS:
            if needle in raw_text:
                _append_unique(positions, english)
        for prefix, ordinal in _ORDINAL_PATTERNS:
            if f"{prefix}\u884c" in raw_text:
                _append_unique(positions, f"{ordinal} row")
            if f"{prefix}\u5217" in raw_text:
                _append_unique(positions, f"{ordinal} column")
        if "\u4e00\u884c" in raw_text and not any("row" in item for item in positions):
            _append_unique(positions, "row")
        if "\u4e00\u5217" in raw_text and not any("column" in item for item in positions):
            _append_unique(positions, "column")
    return positions


def _extract_relations(raw_text: str, category_name: str) -> List[str]:
    category = canonicalize_category_name(category_name)
    relations: List[str] = []
    for needle, english in _RELATION_PATTERNS:
        if needle not in raw_text:
            continue
        if needle == "\u6e38\u6cf3\u6c60" and category == "pool":
            continue
        if needle == "\u5149\u4f0f\u53d1\u7535\u677f" and category == "photovolatic":
            continue
        if needle == "\u96c6\u88c5\u7bb1" and category == "container":
            continue
        _append_unique(relations, english)
    return relations


def build_prompt_spec(raw_text: str, category_name: str) -> PromptSpec:
    category = canonicalize_category_name(category_name)
    normalized_raw_text = " ".join((raw_text or "").strip().lower().split())
    target_phrases = _get_target_phrases(category, normalized_raw_text)
    position_phrases = tuple(_extract_positions(normalized_raw_text))
    if not contains_cjk(normalized_raw_text):
        return PromptSpec(
            prompt=normalized_raw_text,
            target_phrases=target_phrases,
            position_phrases=position_phrases,
        )

    prompt_parts: List[str] = []
    for item in _category_prompt(category):
        _append_unique(prompt_parts, item)
    for item in _extract_descriptors(normalized_raw_text):
        _append_unique(prompt_parts, item)
    for item in _extract_relations(normalized_raw_text, category):
        _append_unique(prompt_parts, item)

    for item in position_phrases:
        _append_unique(prompt_parts, item)

    prompt = " ".join(prompt_parts) if prompt_parts else category
    return PromptSpec(
        prompt=prompt,
        target_phrases=target_phrases,
        position_phrases=tuple(position_phrases),
    )
