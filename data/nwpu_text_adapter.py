from dataclasses import dataclass
import re
from typing import List, Tuple


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

_DESCRIPTOR_PATTERNS = [
    ("所有", "all"),
    ("最大", "largest"),
    ("最小", "smallest"),
    ("白色", "white"),
    ("黑色", "black"),
    ("红色", "red"),
    ("蓝色", "blue"),
    ("黄色", "yellow"),
    ("绿色", "green"),
    ("棕色", "brown"),
    ("橙红色", "orange red"),
    ("大货车", "large truck"),
    ("大汽车", "large car"),
    ("公交车", "bus"),
]

_POSITION_PATTERNS = [
    ("左边", "left"),
    ("右边", "right"),
    ("上面", "top"),
    ("上方", "top"),
    ("下面", "bottom"),
    ("下方", "bottom"),
    ("中间", "center"),
    ("道路上", "on road"),
    ("道路上的", "on road"),
    ("河流上", "on river"),
    ("海面上", "on sea"),
    ("海上", "on sea"),
    ("停车场内", "in parking"),
    ("跑道上", "on runway"),
    ("跑道上的", "on runway"),
    ("屋顶", "on roof"),
    ("附近", "nearby"),
]

_RELATION_PATTERNS = [
    ("游泳池", "with pool"),
    ("光伏发电板", "with photovolatic"),
    ("集装箱", "with container"),
]

_ORDINAL_PATTERNS = [
    ("第一", "first"),
    ("第二", "second"),
    ("第三", "third"),
    ("第四", "fourth"),
    ("第五", "fifth"),
    ("第六", "sixth"),
    ("第七", "seventh"),
    ("第八", "eighth"),
    ("第九", "ninth"),
    ("第十", "tenth"),
]


def canonicalize_category_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text or ""))


def _append_unique(items: List[str], value: str) -> None:
    value = " ".join(value.split())
    if value and value not in items:
        items.append(value)


def _category_prompt(category_name: str) -> List[str]:
    category = canonicalize_category_name(category_name)
    parts = [category]
    for alias in _CATEGORY_ALIASES.get(category, ()):  # keep the exact dataset label first
        _append_unique(parts, alias)
    return parts


def _extract_descriptors(raw_text: str) -> List[str]:
    descriptors: List[str] = []
    for needle, english in _DESCRIPTOR_PATTERNS:
        if needle in raw_text:
            _append_unique(descriptors, english)
    if "行驶" in raw_text:
        _append_unique(descriptors, "moving")
    if "向左行驶" in raw_text:
        _append_unique(descriptors, "moving left")
    if "向右行驶" in raw_text:
        _append_unique(descriptors, "moving right")
    if "向上行驶" in raw_text:
        _append_unique(descriptors, "moving up")
    if "向下行驶" in raw_text:
        _append_unique(descriptors, "moving down")
    return descriptors


def _extract_positions(raw_text: str) -> List[str]:
    positions: List[str] = []
    for needle, english in _POSITION_PATTERNS:
        if needle in raw_text:
            _append_unique(positions, english)
    for prefix, ordinal in _ORDINAL_PATTERNS:
        if f"{prefix}行" in raw_text:
            _append_unique(positions, f"{ordinal} row")
        if f"{prefix}列" in raw_text:
            _append_unique(positions, f"{ordinal} column")
    if "一行" in raw_text and not any("row" in item for item in positions):
        _append_unique(positions, "row")
    if "一列" in raw_text and not any("column" in item for item in positions):
        _append_unique(positions, "column")
    return positions


def _extract_relations(raw_text: str, category_name: str) -> List[str]:
    category = canonicalize_category_name(category_name)
    relations: List[str] = []
    for needle, english in _RELATION_PATTERNS:
        if needle not in raw_text:
            continue
        if needle == "游泳池" and category == "pool":
            continue
        if needle == "光伏发电板" and category == "photovolatic":
            continue
        if needle == "集装箱" and category == "container":
            continue
        _append_unique(relations, english)
    return relations


def build_prompt_spec(raw_text: str, category_name: str) -> PromptSpec:
    category = canonicalize_category_name(category_name)
    normalized_raw_text = " ".join((raw_text or "").strip().lower().split())
    if not contains_cjk(normalized_raw_text):
        return PromptSpec(prompt=normalized_raw_text, target_phrases=(category,), position_phrases=tuple())

    prompt_parts: List[str] = []
    for item in _category_prompt(category):
        _append_unique(prompt_parts, item)
    for item in _extract_descriptors(normalized_raw_text):
        _append_unique(prompt_parts, item)
    for item in _extract_relations(normalized_raw_text, category):
        _append_unique(prompt_parts, item)
    position_phrases = _extract_positions(normalized_raw_text)
    for item in position_phrases:
        _append_unique(prompt_parts, item)

    prompt = " ".join(prompt_parts) if prompt_parts else category
    return PromptSpec(prompt=prompt, target_phrases=(category,), position_phrases=tuple(position_phrases))
