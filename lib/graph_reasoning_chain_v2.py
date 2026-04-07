import math
from typing import Any, Dict, List, Optional, Tuple
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_LANG_ENGLISH_ID = 0
TEXT_LANG_CHINESE_ID = 1

try:
    from bert.tokenization_bert import BertTokenizer
except Exception:
    BertTokenizer = None

from .rs_vocabulary import get_lexicon_for_dataset


class FineGrainedTextParser:
    """Fine-grained parser for nested expressions.

    The parser extracts entities/attributes/relations and also produces
    step-aware reasoning units for progressive graph reasoning.
    """

    def __init__(
        self,
        tokenizer_name: Optional[str] = None,
        tokenizer_name_zh: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        self.tokenizer = None
        if BertTokenizer is not None and tokenizer_name is not None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                self.tokenizer = None

        self.tokenizer_zh = None
        zh_name = tokenizer_name_zh or tokenizer_name
        if BertTokenizer is not None and zh_name is not None:
            try:
                self.tokenizer_zh = BertTokenizer.from_pretrained(zh_name)
            except Exception:
                self.tokenizer_zh = None

        lexicon = get_lexicon_for_dataset(dataset_name)
        self.entity_lexicon = set(lexicon["entities"])
        self.attr_lexicon = set(lexicon["attributes"])
        self.relation_lexicon = set(lexicon["relations"])

        self.reference_relations = {
            "next to", "beside", "near", "close to",
            "above", "below", "over", "under",
            "left", "right", "left of", "right of",
            "in front", "in front of", "behind", "back",
            "inside", "outside", "within", "around",
            "between", "among", "adjacent", "opposite",
        }
        self.modification_relations = {
            "on", "on top of", "upon",
            "in", "inside",
            "at", "by",
            "across", "through", "along", "throughout",
        }

        # Chinese lexicon maps raw Chinese terms to canonical English labels used by downstream graph logic.
        entity_zh_map = lexicon.get("entity_zh_map") or {
            "汽车": "car", "车辆": "car", "船": "ship", "舰": "ship", "飞机": "airplane",
            "道路": "road", "公路": "road", "桥": "bridge", "建筑": "building", "停车场": "parking",
        }
        attr_zh_map = lexicon.get("attribute_zh_map") or {
            "最大": "largest", "最小": "smallest", "白色": "white", "黑色": "black", "红色": "red", "蓝色": "blue",
        }
        rel_zh_map = lexicon.get("relation_zh_map") or {
            "左侧": ("left", "reference"), "右侧": ("right", "reference"),
            "上方": ("top", "reference"), "下方": ("bottom", "reference"),
            "附近": ("near", "reference"), "在": ("in", "modification"),
        }

        self.entity_zh_terms: List[Tuple[str, str]] = list(entity_zh_map.items())
        self.attr_zh_terms: List[Tuple[str, str]] = list(attr_zh_map.items())
        self.rel_zh_terms: List[Tuple[str, str, str]] = [
            (zh, canon, rel_type) for zh, (canon, rel_type) in rel_zh_map.items()
        ]

    @staticmethod
    def _strip_token(token: str) -> str:
        return token.lower().replace("##", "")

    @staticmethod
    def _strip_token_surface(token: str) -> str:
        tok = token
        for prefix in ("##", "Ġ", "▁"):
            if tok.startswith(prefix):
                tok = tok[len(prefix):]
        if tok in {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>"}:
            return ""
        return tok

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    def _build_joined_text_with_spans(self, tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
        pieces: List[str] = []
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            surf = self._strip_token_surface(tok)
            if surf:
                start = cursor
                cursor += len(surf)
                end = cursor
                pieces.append(surf)
                spans.append((start, end))
            else:
                spans.append((cursor, cursor))
        return "".join(pieces), spans

    @staticmethod
    def _char_span_to_token_span(token_char_spans: List[Tuple[int, int]], char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
        s_idx = None
        e_idx = None
        for i, (s, e) in enumerate(token_char_spans):
            if e <= s:
                continue
            if s_idx is None and e > char_start:
                s_idx = i
            if s < char_end:
                e_idx = i
        if s_idx is None or e_idx is None or e_idx < s_idx:
            return None
        return s_idx, e_idx + 1

    def _find_chinese_matches(self, joined_text: str, token_char_spans: List[Tuple[int, int]], term_map: List[Tuple[str, str]]) -> List[Tuple[str, Tuple[int, int]]]:
        matches: List[Tuple[str, Tuple[int, int]]] = []
        for needle, canonical in term_map:
            start = 0
            while True:
                idx = joined_text.find(needle, start)
                if idx < 0:
                    break
                span = self._char_span_to_token_span(token_char_spans, idx, idx + len(needle))
                if span is not None:
                    matches.append((canonical, span))
                start = idx + len(needle)
        matches.sort(key=lambda x: (x[1][0], x[1][1]))
        return matches

    def _find_chinese_relations(
        self,
        joined_text: str,
        token_char_spans: List[Tuple[int, int]],
    ) -> List[Tuple[str, Tuple[int, int], str]]:
        relations: List[Tuple[str, Tuple[int, int], str]] = []
        for needle, canonical, rel_type in self.rel_zh_terms:
            start = 0
            while True:
                idx = joined_text.find(needle, start)
                if idx < 0:
                    break
                span = self._char_span_to_token_span(token_char_spans, idx, idx + len(needle))
                if span is not None:
                    item = (canonical, span, rel_type)
                    if not any(r[0] == item[0] and r[1] == item[1] for r in relations):
                        relations.append(item)
                start = idx + len(needle)
        relations.sort(key=lambda x: (x[1][0], x[1][1]))
        return relations

    def _parse_single(
        self,
        input_ids: torch.Tensor,
        valid_len: int,
        prefer_chinese: bool,
    ) -> Dict[str, Any]:
        tokenizer = self.tokenizer_zh if prefer_chinese and self.tokenizer_zh is not None else self.tokenizer
        if tokenizer is None:
            return {
                "target": {"word": None, "token_span": None},
                "entities": [],
                "attributes": [],
                "relations": [],
                "tokens": [],
                "entity_attributes": {},
                "spatial_triplets": [],
                "entity_tokens_map": {},
                "attr_tokens_map": {},
                "rel_tokens_map": {},
                "reasoning_units": [(0, max(1, valid_len))],
                "valid_len": max(1, valid_len),
            }

        ids = input_ids.detach().cpu().tolist()[:valid_len]
        tokens = tokenizer.convert_ids_to_tokens(ids)

        if prefer_chinese:
            joined_text, token_char_spans = self._build_joined_text_with_spans(tokens)
            chinese_like = self._contains_cjk(joined_text)
        else:
            joined_text, token_char_spans = "", []
            chinese_like = False

        if prefer_chinese and chinese_like:
            entities = self._find_chinese_matches(joined_text, token_char_spans, self.entity_zh_terms)
            attributes = self._find_chinese_matches(joined_text, token_char_spans, self.attr_zh_terms)
            relations = self._find_chinese_relations(joined_text, token_char_spans)
        else:
            entities = self._find_components(tokens, self.entity_lexicon)
            attributes = self._find_components(tokens, self.attr_lexicon)
            relations = self._find_relations_with_type(tokens)

        target = entities[-1] if entities else None
        entity_attributes = self._build_entity_attribute_map(entities, attributes)
        spatial_triplets = self._extract_spatial_triplets(entities, relations, target)
        reasoning_units = self._build_reasoning_units(entities, attributes, relations, valid_len)

        return {
            "target": {
                "word": target[0] if target else None,
                "token_span": target[1] if target else None,
            },
            "entities": [
                {
                    "word": ew,
                    "token_span": sp,
                    "type": "target" if target is not None and ew == target[0] and sp == target[1] else "reference",
                    "attributes": entity_attributes.get(ew, []),
                }
                for ew, sp in entities
            ],
            "attributes": [{"word": aw, "token_span": sp} for aw, sp in attributes],
            "relations": [{"word": rw, "token_span": sp, "relation_type": rt} for rw, sp, rt in relations],
            "tokens": tokens,
            "entity_attributes": entity_attributes,
            "spatial_triplets": spatial_triplets,
            "entity_tokens_map": self._build_tokens_map(entities),
            "attr_tokens_map": self._build_tokens_map(attributes),
            "rel_tokens_map": self._build_tokens_map([(r[0], r[1]) for r in relations]),
            "reasoning_units": reasoning_units,
            "valid_len": valid_len,
        }

    def _find_components(self, tokens: List[str], lexicon: set) -> List[Tuple[str, Tuple[int, int]]]:
        components: List[Tuple[str, Tuple[int, int]]] = []
        T = len(tokens)
        for phrase in lexicon:
            p_toks = phrase.lower().split()
            L = len(p_toks)
            if L == 0:
                continue
            for i in range(T - L + 1):
                sl = [self._strip_token(t) for t in tokens[i:i + L]]
                if sl == p_toks:
                    components.append((phrase, (i, i + L)))
        components.sort(key=lambda x: (x[1][0], x[1][1]))
        return components

    def _find_relations_with_type(self, tokens: List[str]) -> List[Tuple[str, Tuple[int, int], str]]:
        relations: List[Tuple[str, Tuple[int, int], str]] = []

        def add_rel(lex: set, rel_type: str):
            T = len(tokens)
            for phrase in lex:
                p_toks = phrase.lower().split()
                L = len(p_toks)
                if L == 0:
                    continue
                for i in range(T - L + 1):
                    sl = [self._strip_token(t) for t in tokens[i:i + L]]
                    if sl == p_toks:
                        item = (phrase, (i, i + L), rel_type)
                        if not any(r[0] == item[0] and r[1] == item[1] for r in relations):
                            relations.append(item)

        add_rel(self.reference_relations, "reference")
        add_rel(self.modification_relations, "modification")
        add_rel(self.relation_lexicon, "spatial")

        relations.sort(key=lambda x: (x[1][0], x[1][1]))
        return relations

    def _build_entity_attribute_map(
        self,
        entities: List[Tuple[str, Tuple[int, int]]],
        attributes: List[Tuple[str, Tuple[int, int]]],
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {e[0]: [] for e in entities}
        for attr_word, (a_s, a_e) in attributes:
            best_entity = None
            best_dist = 1e9

            for e_word, (e_s, _) in entities:
                if e_s >= a_e:
                    dist = e_s - a_e
                    if dist < best_dist:
                        best_dist = dist
                        best_entity = e_word

            if best_entity is None:
                for e_word, (_, e_e) in reversed(entities):
                    if e_e <= a_s:
                        dist = a_s - e_e
                        if dist < best_dist:
                            best_dist = dist
                            best_entity = e_word

            if best_entity is not None:
                out.setdefault(best_entity, []).append(attr_word)
        return out

    def _extract_spatial_triplets(
        self,
        entities: List[Tuple[str, Tuple[int, int]]],
        relations: List[Tuple[str, Tuple[int, int], str]],
        target: Optional[Tuple[str, Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        triplets: List[Dict[str, Any]] = []
        for rel_word, (r_s, r_e), rel_type in relations:
            ref = None
            tgt = None

            for e_word, (e_s, e_e) in reversed(entities):
                if e_e <= r_s:
                    ref = (e_word, e_s, e_e)
                    break
            for e_word, (e_s, e_e) in entities:
                if e_s >= r_e:
                    tgt = (e_word, e_s, e_e)
                    break

            if tgt is None and target is not None:
                tgt = (target[0], target[1][0], target[1][1])

            if ref is not None and tgt is not None:
                triplets.append(
                    {
                        "reference": {"word": ref[0], "token_span": (ref[1], ref[2])},
                        "relation": rel_word,
                        "relation_type": rel_type,
                        "target": {"word": tgt[0], "token_span": (tgt[1], tgt[2])},
                        "rel_token_span": (r_s, r_e),
                    }
                )
        return triplets

    @staticmethod
    def _build_tokens_map(components: List[Tuple[str, Tuple[int, int]]]) -> Dict[str, List[Tuple[int, int]]]:
        out: Dict[str, List[Tuple[int, int]]] = {}
        for word, span in components:
            out.setdefault(word, []).append(span)
        return out

    def _build_reasoning_units(
        self,
        entities: List[Tuple[str, Tuple[int, int]]],
        attributes: List[Tuple[str, Tuple[int, int]]],
        relations: List[Tuple[str, Tuple[int, int], str]],
        valid_len: int,
    ) -> List[Tuple[int, int]]:
        # Progressive nested units guided by noun chunks:
        # T = number of entities (bounded later by GRL max steps).
        if len(entities) == 0:
            return [(0, max(1, valid_len))]

        units: List[Tuple[int, int]] = []
        for _, (e_s, e_e) in entities:
            # Include local attributes/relations around the anchor entity.
            l = e_s
            r = e_e
            for _, (a_s, a_e) in attributes:
                if abs(a_e - e_s) <= 3 or abs(a_s - e_e) <= 3:
                    l = min(l, a_s)
                    r = max(r, a_e)
            for _, (rel_s, rel_e), _ in relations:
                if abs(rel_e - e_s) <= 4 or abs(rel_s - e_e) <= 4:
                    l = min(l, rel_s)
                    r = max(r, rel_e)
            units.append((max(0, l), min(valid_len, max(r, l + 1))))

        # Make units progressive and nested by cumulative merge.
        merged: List[Tuple[int, int]] = []
        cur_l, cur_r = units[0]
        merged.append((cur_l, cur_r))
        for l, r in units[1:]:
            cur_l = min(cur_l, l)
            cur_r = max(cur_r, r)
            merged.append((cur_l, cur_r))
        return merged

    def parse_batch_detailed(
        self,
        input_ids: Optional[torch.Tensor],
        l_mask: Optional[torch.Tensor],
        text_lang_ids: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        if l_mask is None:
            raise ValueError("l_mask is required for text parsing")

        if l_mask.dim() == 3:
            l_mask_2d = l_mask.squeeze(-1)
        else:
            l_mask_2d = l_mask

        B, N = l_mask_2d.shape
        if text_lang_ids is not None:
            if text_lang_ids.dim() > 1:
                lang_ids = text_lang_ids.view(text_lang_ids.size(0), -1)[:, 0]
            else:
                lang_ids = text_lang_ids
            lang_ids = lang_ids.detach().cpu().long().view(-1)
        else:
            lang_ids = torch.full((B,), TEXT_LANG_ENGLISH_ID, dtype=torch.long)

        if lang_ids.numel() < B:
            padded = torch.full((B,), TEXT_LANG_ENGLISH_ID, dtype=torch.long)
            padded[:lang_ids.numel()] = lang_ids
            lang_ids = padded
        elif lang_ids.numel() > B:
            lang_ids = lang_ids[:B]

        results: List[Dict[str, Any]] = []

        for b in range(B):
            valid_len = int(l_mask_2d[b].float().sum().item())
            valid_len = max(1, min(valid_len, N))

            if input_ids is None:
                results.append(
                    {
                        "target": {"word": None, "token_span": None},
                        "entities": [],
                        "attributes": [],
                        "relations": [],
                        "tokens": [],
                        "entity_attributes": {},
                        "spatial_triplets": [],
                        "entity_tokens_map": {},
                        "attr_tokens_map": {},
                        "rel_tokens_map": {},
                        "reasoning_units": [(0, valid_len)],
                        "valid_len": valid_len,
                    }
                )
                continue
            prefer_chinese = int(lang_ids[b].item()) == TEXT_LANG_CHINESE_ID
            results.append(self._parse_single(input_ids[b], valid_len, prefer_chinese))

        return results

    # Backward-compatible single-sample API.
    def parse_detailed(self, input_ids: Optional[torch.Tensor], l_mask: Optional[torch.Tensor]) -> Dict[str, Any]:
        return self.parse_batch_detailed(input_ids, l_mask)[0]


class GraphAttentionLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.masked_fill(adj <= 0, -1e4)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)
        return self.proj(out)


class GraphConvLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        agg = torch.matmul(adj / deg, x)
        out = self.linear(agg)
        return self.norm(out)


class GraphReasoningChain(nn.Module):
    """Graph reasoning chain for complex nested text in segmentation.

    Upgrades in this version:
    1) Structured text edges from entity-attribute-relation triplets.
    2) Dynamic node/edge gating guided by step-wise sub-expressions.
    3) Adaptive reasoning steps based on parsed noun units.
    4) Dual-graph fusion (main graph + semantic similarity graph).
    5) Segmentation-adapted refinement head (no bbox branch).
    """

    def __init__(
        self,
        in_channels: int,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_visual_nodes: int = 64,
        num_reasoning_steps: int = 2,
        num_steps: Optional[int] = None,
        dropout: float = 0.1,
        residual_scale: float = 0.2,
        residual_clip: float = 1.0,
        grl_mode: str = "full",
        tokenizer_name: Optional[str] = None,
        tokenizer_name_zh: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        super().__init__()

        # Backward compatibility with backbone argument name.
        if num_steps is not None:
            num_reasoning_steps = num_steps

        self.num_visual_nodes = num_visual_nodes
        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = max(1, num_reasoning_steps)
        self.num_relation_types = 14  # 11 geo relations + 3 scale relations
        self.multiscale_strides = (1, 2, 4)
        valid_modes = {"full", "no_parser", "off"}
        if grl_mode not in valid_modes:
            raise ValueError(f"Unsupported grl_mode={grl_mode}. Expected one of {sorted(valid_modes)}")
        self.grl_mode = grl_mode
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)

        self.visual_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.text_proj = nn.Conv1d(text_dim, hidden_dim, kernel_size=1)

        self.entity_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attr_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.entity_score = nn.Linear(hidden_dim, 1, bias=False)
        self.rel_gate = nn.Linear(hidden_dim, self.num_relation_types, bias=True)
        self.rel_text_gate = nn.Linear(hidden_dim, self.num_relation_types, bias=False)
        self.edge_update_alpha = nn.Parameter(torch.tensor(0.10))

        # Learned noise prior map for remote-sensing clutter suppression.
        self.noise_prior_head = nn.Sequential(
            nn.Conv2d(hidden_dim, max(hidden_dim // 4, 32), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(hidden_dim // 4, 32)),
            nn.GELU(),
            nn.Conv2d(max(hidden_dim // 4, 32), 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.gat_layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim, dropout=dropout) for _ in range(self.num_reasoning_steps)]
        )
        self.gcn_layers = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(self.num_reasoning_steps)])

        # Dual-graph branch for semantic similarity graph.
        self.cls_gat_layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim, dropout=dropout) for _ in range(self.num_reasoning_steps)]
        )
        self.cls_gcn_layers = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(self.num_reasoning_steps)])

        self.reason_norm = nn.LayerNorm(hidden_dim)
        self.graph_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.reconstruct = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # Segmentation-adapted refinement head.
        half = max(hidden_dim // 2, 32)
        self.seg_refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, half, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.reconstruct[0].weight)
        nn.init.zeros_(self.reconstruct[1].weight)
        nn.init.zeros_(self.reconstruct[1].bias)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.refine_alpha = nn.Parameter(torch.tensor(0.0))

        self.text_parser = FineGrainedTextParser(
            tokenizer_name=tokenizer_name,
            tokenizer_name_zh=tokenizer_name_zh,
            dataset_name=dataset_name,
        )

    @staticmethod
    def _build_unparsed_batch(l_mask: torch.Tensor) -> List[Dict[str, Any]]:
        if l_mask.dim() == 3:
            l_mask_2d = l_mask.squeeze(-1)
        else:
            l_mask_2d = l_mask

        batch: List[Dict[str, Any]] = []
        B, N = l_mask_2d.shape
        for b in range(B):
            valid_len = int(l_mask_2d[b].float().sum().item())
            valid_len = max(1, min(valid_len, N))
            batch.append(
                {
                    "target": {"word": None, "token_span": None},
                    "entities": [],
                    "attributes": [],
                    "relations": [],
                    "tokens": [],
                    "entity_attributes": {},
                    "spatial_triplets": [],
                    "entity_tokens_map": {},
                    "attr_tokens_map": {},
                    "rel_tokens_map": {},
                    "reasoning_units": [(0, valid_len)],
                    "valid_len": valid_len,
                }
            )
        return batch

    @staticmethod
    def _normalize_text_lang_ids(text_lang_ids: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if text_lang_ids is None:
            return torch.full((batch_size,), TEXT_LANG_ENGLISH_ID, dtype=torch.long, device=device)

        if text_lang_ids.dim() > 1:
            lang_ids = text_lang_ids.view(text_lang_ids.size(0), -1)[:, 0]
        else:
            lang_ids = text_lang_ids

        lang_ids = lang_ids.to(device=device, dtype=torch.long).view(-1)
        if lang_ids.numel() < batch_size:
            padded = torch.full((batch_size,), TEXT_LANG_ENGLISH_ID, dtype=torch.long, device=device)
            padded[:lang_ids.numel()] = lang_ids
            return padded
        if lang_ids.numel() > batch_size:
            return lang_ids[:batch_size]
        return lang_ids

    def _build_routed_parse_batch(
        self,
        text_ids: Optional[torch.Tensor],
        l_mask: torch.Tensor,
        text_lang_ids: Optional[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        if l_mask.dim() == 3:
            l_mask_2d = l_mask.squeeze(-1)
        else:
            l_mask_2d = l_mask

        batch_size = l_mask_2d.size(0)
        if text_ids is None:
            return self._build_unparsed_batch(l_mask_2d)

        lang_ids = self._normalize_text_lang_ids(text_lang_ids, batch_size, l_mask_2d.device)
        parse_batch: List[Dict[str, Any]] = []

        for b in range(batch_size):
            sample_text_ids = text_ids[b:b + 1]
            sample_mask = l_mask_2d[b:b + 1]
            lang_id = int(lang_ids[b].item())

            lang_tensor = torch.tensor([lang_id], dtype=torch.long)
            parse_result = self.text_parser.parse_batch_detailed(
                sample_text_ids,
                sample_mask,
                text_lang_ids=lang_tensor,
            )[0]

            parse_batch.append(parse_result)

        return parse_batch

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        w = mask.float().clamp(min=0.0, max=1.0)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (x * w).sum(dim=1) / denom

    def _build_step_queries(
        self,
        txt: torch.Tensor,
        parse_result: Dict[str, Any],
        max_steps: int,
    ) -> torch.Tensor:
        # txt: [1, N, D]
        N = txt.shape[1]
        D = txt.shape[2]
        device = txt.device

        units = parse_result.get("reasoning_units", [(0, max(1, parse_result.get("valid_len", N)))])
        if len(units) == 0:
            units = [(0, max(1, parse_result.get("valid_len", N)))]

        # Adaptive number of steps from noun-phrase units.
        T = min(max_steps, len(units))
        units = units[:T]

        queries: List[torch.Tensor] = []
        for (s, e) in units:
            s = max(0, min(int(s), N - 1))
            e = max(s + 1, min(int(e), N))
            q = txt[:, s:e, :].mean(dim=1)  # [1, D]
            queries.append(q)

        if len(queries) == 0:
            queries = [txt.mean(dim=1)]

        return torch.stack(queries, dim=1)  # [1, T, D]

    def _build_similarity_adjacency(self, nodes: torch.Tensor) -> torch.Tensor:
        # nodes: [1, N, D]
        n = F.normalize(nodes, dim=-1)
        sim = torch.matmul(n, n.transpose(-1, -2))
        sim = (sim + 1.0) * 0.5
        sim = sim.clamp(0.0, 1.0)
        eye = torch.eye(sim.shape[-1], device=sim.device).unsqueeze(0)
        return (0.15 * sim + eye).clamp(0.0, 1.0)

    def _build_multiscale_candidates(
        self,
        vis_map: torch.Tensor,
        noise_map: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # vis_map: [1, D, H, W], noise_map: [1, 1, H, W]
        node_list: List[torch.Tensor] = []
        coord_list: List[torch.Tensor] = []
        scale_list: List[torch.Tensor] = []
        noise_list: List[torch.Tensor] = []

        for level, stride in enumerate(self.multiscale_strides, start=1):
            h_s = max(1, H // stride)
            w_s = max(1, W // stride)

            feat_s = F.adaptive_avg_pool2d(vis_map, output_size=(h_s, w_s))
            noi_s = F.adaptive_avg_pool2d(noise_map, output_size=(h_s, w_s))

            nodes = feat_s.flatten(2).transpose(1, 2)  # [1, h_s*w_s, D]
            noise = noi_s.flatten(2).transpose(1, 2).squeeze(-1)  # [1, h_s*w_s]

            yy = torch.arange(h_s, device=vis_map.device).float()
            xx = torch.arange(w_s, device=vis_map.device).float()
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")

            y_img = ((yy + 0.5) * stride - 0.5).clamp(0.0, max(float(H - 1), 1.0))
            x_img = ((xx + 0.5) * stride - 0.5).clamp(0.0, max(float(W - 1), 1.0))

            y_norm = y_img / max(float(H - 1), 1.0)
            x_norm = x_img / max(float(W - 1), 1.0)
            coords = torch.stack([x_norm, y_norm], dim=-1).view(1, h_s * w_s, 2)

            scales = torch.full((1, h_s * w_s), float(level), device=vis_map.device)

            node_list.append(nodes)
            coord_list.append(coords)
            scale_list.append(scales)
            noise_list.append(noise)

        cand_nodes = torch.cat(node_list, dim=1)
        cand_coords = torch.cat(coord_list, dim=1)
        cand_scales = torch.cat(scale_list, dim=1)
        cand_noise = torch.cat(noise_list, dim=1)
        return cand_nodes, cand_coords, cand_scales, cand_noise

    def _relation_text_summary(
        self,
        txt: torch.Tensor,
        parse_result: Dict[str, Any],
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        rel_spans = []
        for tri in parse_result.get("spatial_triplets", []):
            span = tri.get("rel_token_span")
            if span is not None:
                rel_spans.append(span)

        if len(rel_spans) == 0:
            return fallback

        rel_feats: List[torch.Tensor] = []
        N = txt.shape[1]
        for s, e in rel_spans:
            s = max(0, min(int(s), N - 1))
            e = max(s + 1, min(int(e), N))
            rel_feats.append(txt[:, s:e, :].mean(dim=1))

        return torch.stack(rel_feats, dim=1).mean(dim=1)

    def _inject_node_updates(
        self,
        base_map: torch.Tensor,
        node_delta: torch.Tensor,
        coords: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        # base_map: [1, D, H, W], node_delta: [1, K, D], coords: [1, K, 2]
        out = base_map.clone()
        weight = torch.zeros(1, 1, H, W, device=base_map.device)

        px = (coords[0, :, 0] * max(float(W - 1), 1.0)).round().long().clamp(0, W - 1)
        py = (coords[0, :, 1] * max(float(H - 1), 1.0)).round().long().clamp(0, H - 1)

        K = node_delta.shape[1]
        for k in range(K):
            out[0, :, py[k], px[k]] = out[0, :, py[k], px[k]] + node_delta[0, k, :]
            weight[0, 0, py[k], px[k]] = weight[0, 0, py[k], px[k]] + 1.0

        return out / (1.0 + weight)

    def _build_text_graph_single(
        self,
        text_embeddings: torch.Tensor,  # [1, N, D]
        parse_result: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = text_embeddings.device
        nodes: List[torch.Tensor] = []

        # Keep target at index 0 for downstream selection.
        target_span = parse_result.get("target", {}).get("token_span")
        if target_span is not None:
            ts, te = target_span
            target_emb = text_embeddings[:, ts:te, :].mean(dim=1)
        else:
            target_emb = text_embeddings.mean(dim=1)
        nodes.append(self.entity_proj(target_emb).unsqueeze(1))

        # Index maps.
        ref_idx: Dict[str, int] = {}
        attr_idx: Dict[Tuple[str, str], int] = {}
        rel_nodes: List[Tuple[int, int, int]] = []

        # Reference entities.
        for ent in parse_result.get("entities", []):
            if ent.get("type") != "reference":
                continue
            span = ent.get("token_span")
            if span is None:
                continue
            es, ee = span
            emb = text_embeddings[:, es:ee, :].mean(dim=1)
            idx = len(nodes)
            ref_idx[ent.get("word", f"ref_{idx}")] = idx
            nodes.append(self.entity_proj(emb).unsqueeze(1))

        # Attributes linked to entities.
        attr_used = set()
        for ent_word, attrs in parse_result.get("entity_attributes", {}).items():
            for attr_word in attrs:
                if (ent_word, attr_word) in attr_used:
                    continue
                attr_used.add((ent_word, attr_word))
                found = None
                for attr in parse_result.get("attributes", []):
                    if attr.get("word") == attr_word:
                        found = attr.get("token_span")
                        break
                if found is None:
                    continue
                as_, ae = found
                emb = text_embeddings[:, as_:ae, :].mean(dim=1)
                idx = len(nodes)
                attr_idx[(ent_word, attr_word)] = idx
                nodes.append(self.attr_proj(emb).unsqueeze(1))

        # Relation nodes from triplets.
        for tri in parse_result.get("spatial_triplets", []):
            r_span = tri.get("rel_token_span")
            if r_span is None:
                continue
            rs, re = r_span
            emb = text_embeddings[:, rs:re, :].mean(dim=1)
            rel_index = len(nodes)
            nodes.append(self.rel_proj(emb).unsqueeze(1))

            ref_word = tri.get("reference", {}).get("word")
            ref_node = ref_idx.get(ref_word, None)
            tgt_node = 0
            rel_nodes.append((ref_node if ref_node is not None else tgt_node, rel_index, tgt_node))

        text_nodes = torch.cat(nodes, dim=1)  # [1, M, D]
        M = text_nodes.shape[1]

        # Structured adjacency.
        A = torch.zeros(1, M, M, device=device)
        A[:, torch.arange(M), torch.arange(M)] = 1.0

        # Target <-> references.
        for _, idx in ref_idx.items():
            A[:, 0, idx] = 1.0
            A[:, idx, 0] = 1.0

        # Entity <-> attribute.
        for (ent_word, _), a_idx in attr_idx.items():
            e_idx = ref_idx.get(ent_word, 0)
            A[:, e_idx, a_idx] = 1.0
            A[:, a_idx, e_idx] = 1.0

        # Triplet chain: reference <-> relation <-> target.
        for ref_node, rel_node, tgt_node in rel_nodes:
            A[:, ref_node, rel_node] = 1.0
            A[:, rel_node, ref_node] = 1.0
            A[:, rel_node, tgt_node] = 1.0
            A[:, tgt_node, rel_node] = 1.0
            A[:, ref_node, tgt_node] = torch.maximum(A[:, ref_node, tgt_node], torch.tensor(0.5, device=device))
            A[:, tgt_node, ref_node] = torch.maximum(A[:, tgt_node, ref_node], torch.tensor(0.5, device=device))

        # Keep graph connected with a weak prior.
        A = torch.maximum(A, 0.05 * torch.ones_like(A))
        return text_nodes, A

    def _build_visual_adjacency(
        self,
        coords: torch.Tensor,
        rel_weights: torch.Tensor,
        scale_levels: torch.Tensor,
    ) -> torch.Tensor:
        # coords: [1, K, 2], rel_weights: [1, 14], scale_levels: [1, K]
        _, K, _ = coords.shape

        xi = coords[:, :, 0].unsqueeze(-1)
        yi = coords[:, :, 1].unsqueeze(-1)
        xj = coords[:, :, 0].unsqueeze(1)
        yj = coords[:, :, 1].unsqueeze(1)

        dx = xj - xi
        dy = yj - yi
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)

        # 8-direction geo edges with angle sectors.
        theta = torch.atan2(-dy, dx)

        def circ_diff(a: torch.Tensor, c: float) -> torch.Tensor:
            return torch.atan2(torch.sin(a - c), torch.cos(a - c)).abs()

        sigma = 0.65
        A_east = torch.exp(-(circ_diff(theta, 0.0) ** 2) / (2 * sigma * sigma))
        A_south = torch.exp(-(circ_diff(theta, -math.pi / 2) ** 2) / (2 * sigma * sigma))
        A_west = torch.exp(-(circ_diff(theta, math.pi) ** 2) / (2 * sigma * sigma))
        A_north = torch.exp(-(circ_diff(theta, math.pi / 2) ** 2) / (2 * sigma * sigma))
        A_se = torch.exp(-(circ_diff(theta, -math.pi / 4) ** 2) / (2 * sigma * sigma))
        A_sw = torch.exp(-(circ_diff(theta, -3 * math.pi / 4) ** 2) / (2 * sigma * sigma))
        A_ne = torch.exp(-(circ_diff(theta, math.pi / 4) ** 2) / (2 * sigma * sigma))
        A_nw = torch.exp(-(circ_diff(theta, 3 * math.pi / 4) ** 2) / (2 * sigma * sigma))

        # 3 topological geo edges: contain / inside / adjacent
        si = scale_levels.unsqueeze(-1)
        sj = scale_levels.unsqueeze(1)
        A_contain = ((si > sj).float() * (dist < 0.35).float())
        A_inside = ((si < sj).float() * (dist < 0.35).float())
        A_adjacent = ((dist < 0.20).float() * (dist > 0.03).float())

        # 3 scale edges: larger / smaller / similar
        dscale = si - sj
        A_larger = (dscale > 0.5).float()
        A_smaller = (dscale < -0.5).float()
        A_similar = (dscale.abs() <= 0.5).float()

        w = F.softmax(rel_weights, dim=-1)
        A = (
            w[:, 0].view(1, 1, 1) * A_east
            + w[:, 1].view(1, 1, 1) * A_south
            + w[:, 2].view(1, 1, 1) * A_west
            + w[:, 3].view(1, 1, 1) * A_north
            + w[:, 4].view(1, 1, 1) * A_se
            + w[:, 5].view(1, 1, 1) * A_sw
            + w[:, 6].view(1, 1, 1) * A_ne
            + w[:, 7].view(1, 1, 1) * A_nw
            + w[:, 8].view(1, 1, 1) * A_contain
            + w[:, 9].view(1, 1, 1) * A_inside
            + w[:, 10].view(1, 1, 1) * A_adjacent
            + w[:, 11].view(1, 1, 1) * A_larger
            + w[:, 12].view(1, 1, 1) * A_smaller
            + w[:, 13].view(1, 1, 1) * A_similar
        )
        eye = torch.eye(K, device=coords.device).unsqueeze(0)
        return A + eye

    def _dynamic_gate(self, nodes: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # nodes: [1, N, D], query: [1, D]
        score = (nodes * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        score = torch.sigmoid(score)
        mean = score.mean(dim=1, keepdim=True)
        keep = (score > 0.85 * mean).float()
        # Keep weak gradients even for filtered nodes.
        gate = 0.1 + 0.9 * (keep * score)
        return gate

    def forward(
        self,
        x: torch.Tensor,
        l: torch.Tensor,
        l_mask: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        text_lang_ids: Optional[torch.Tensor] = None,
        t_mask: Optional[torch.Tensor] = None,
        p_mask: Optional[torch.Tensor] = None,
        text_struct: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # x: [B, C, H, W], l: [B, D_text, N]
        if self.grl_mode == "off":
            return x

        B, _, H, W = x.shape

        vis_map_all = self.visual_proj(x)  # [B, D, H, W]
        vis_all = vis_map_all.flatten(2).transpose(1, 2)  # [B, HW, D]
        txt_all = self.text_proj(l).transpose(1, 2)  # [B, N, D]
        noise_all = self.noise_prior_head(vis_map_all)  # [B, 1, H, W]

        if l_mask.dim() == 2:
            l_mask = l_mask.unsqueeze(-1)

        if self.grl_mode == "no_parser":
            parse_batch = self._build_unparsed_batch(l_mask)
        elif text_struct is not None and isinstance(text_struct, dict) and "batch" in text_struct:
            parse_batch = text_struct["batch"]
        elif text_struct is not None and isinstance(text_struct, dict) and "target" in text_struct:
            parse_batch = [text_struct for _ in range(B)]
        else:
            parse_batch = self._build_routed_parse_batch(text_ids, l_mask, text_lang_ids)

        refined_dense = []
        for b in range(B):
            vis = vis_all[b:b + 1]  # [1, HW, D]
            vis_map = vis_map_all[b:b + 1]  # [1, D, H, W]
            txt = txt_all[b:b + 1]  # [1, N, D]
            noise_map = noise_all[b:b + 1]  # [1, 1, H, W]
            parse_result = parse_batch[b] if b < len(parse_batch) else parse_batch[0]

            text_nodes, A_text = self._build_text_graph_single(txt, parse_result)  # [1, M, D], [1, M, M]
            target_emb = text_nodes[:, 0, :]  # [1, D]

            # Build multi-scale visual candidates and suppress noisy regions.
            cand_nodes, cand_coords, cand_scales, cand_noise = self._build_multiscale_candidates(
                vis_map, noise_map, H, W
            )

            # Visual node selection guided by target + learned score + noise suppression.
            vis_logits = (cand_nodes * target_emb.unsqueeze(1)).sum(dim=-1)
            vis_logits = vis_logits + self.entity_score(cand_nodes).squeeze(-1)
            vis_logits = vis_logits * (1.0 - cand_noise)

            HW = cand_nodes.shape[1]
            K = min(self.num_visual_nodes, HW)
            node_idx = torch.topk(vis_logits, k=K, dim=1, largest=True).indices  # [1, K]
            gather_idx = node_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            vis_nodes = torch.gather(cand_nodes, dim=1, index=gather_idx)  # [1, K, D]

            # Visual graph.
            coord_idx = node_idx.unsqueeze(-1).expand(-1, -1, 2)
            coords = torch.gather(cand_coords, dim=1, index=coord_idx)
            scale_idx = node_idx
            scale_levels = torch.gather(cand_scales, dim=1, index=scale_idx)

            rel_text_summary = self._relation_text_summary(txt, parse_result, target_emb)
            rel_weights = self.rel_gate(target_emb) + self.rel_text_gate(rel_text_summary)
            A_vis = self._build_visual_adjacency(coords, rel_weights, scale_levels)

            # Build multimodal main graph.
            M = text_nodes.shape[1]
            N_all = K + M
            nodes = torch.cat([vis_nodes, text_nodes], dim=1)
            A_main = torch.zeros(1, N_all, N_all, device=x.device)
            A_main[:, :K, :K] = A_vis

            vt = torch.matmul(vis_nodes, text_nodes.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            vt = torch.sigmoid(vt)
            A_main[:, :K, K:] = vt
            A_main[:, K:, :K] = vt.transpose(-1, -2)
            A_main[:, K:, K:] = A_text

            # Build semantic similarity graph (dual graph).
            A_cls = self._build_similarity_adjacency(nodes)

            # Progressive step queries (adaptive chain length).
            step_queries = self._build_step_queries(txt, parse_result, self.num_reasoning_steps)
            T = step_queries.shape[1]

            h = nodes
            for t in range(T):
                q_t = step_queries[:, t, :]  # [1, D]
                gate = self._dynamic_gate(h, q_t)  # [1, N_all]
                edge_gate = gate.unsqueeze(-1) * gate.unsqueeze(-2)

                # Step-wise edge update with relation text signal.
                rel_step = torch.tanh(self.rel_text_gate(q_t))
                rel_weights_t = rel_weights + self.edge_update_alpha * float(t + 1) * rel_step
                A_vis_t = self._build_visual_adjacency(coords, rel_weights_t, scale_levels)

                A_step = A_main.clone()
                A_step[:, :K, :K] = A_vis_t

                A_t_main = A_step * edge_gate
                A_t_cls = A_cls * edge_gate

                idx = t % self.num_reasoning_steps

                h_main = self.reason_norm(h + self.gat_layers[idx](h, A_t_main))
                h_main = self.reason_norm(h_main + self.gcn_layers[idx](h_main, A_t_main))

                h_cls = self.reason_norm(h + self.cls_gat_layers[idx](h, A_t_cls))
                h_cls = self.reason_norm(h_cls + self.cls_gcn_layers[idx](h_cls, A_t_cls))

                fuse = self.graph_fuse(torch.cat([h_main, h_cls], dim=-1))
                h = fuse * h_main + (1.0 - fuse) * h_cls

            vis_refined = h[:, :K, :]
            vis_delta = vis_refined - vis_nodes
            vis_out = self._inject_node_updates(vis_map, vis_delta, coords, H, W)
            refined_dense.append(vis_out)

        vis_refined_map = torch.cat(refined_dense, dim=0)  # [B, D, H, W]

        # Segmentation-adapted refinement weighting.
        seg_gate = self.seg_refine_head(vis_refined_map)
        delta = self.reconstruct(vis_refined_map)
        delta = delta * (1.0 + torch.tanh(self.refine_alpha) * seg_gate)

        residual = torch.tanh(self.alpha) * delta
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        if self.residual_clip > 0:
            residual = residual.clamp(min=-self.residual_clip, max=self.residual_clip)

        return x + self.residual_scale * residual
