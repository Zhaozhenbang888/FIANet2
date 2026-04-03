import os
import json
import pickle
from collections import Counter

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools import mask as mask_utils

from bert.tokenization_bert import BertTokenizer
from data.nwpu_text_adapter import (
    build_prompt_spec,
    canonicalize_category_name,
    classify_text_language,
    text_matches_language_filter,
)


def _resolve_nwpu_paths(data_root):
    image_dir = os.path.join(data_root, "image", "image")
    refs_file = os.path.join(data_root, "new_refs(unc).p")
    instances_file = os.path.join(data_root, "new_instances.json")

    # Keep backward compatibility for old file names.
    if not os.path.exists(refs_file):
        refs_file = os.path.join(data_root, "refs(unc).p")
    if not os.path.exists(instances_file):
        instances_file = os.path.join(data_root, "instances.json")

    return image_dir, refs_file, instances_file


def _load_instances_and_refs(data_root):
    image_dir, refs_file, instances_file = _resolve_nwpu_paths(data_root)

    with open(refs_file, "rb") as rf:
        refs = pickle.load(rf)
    with open(instances_file, "r", encoding="utf-8") as jf:
        instances = json.load(jf)

    images_by_id = {img["id"]: img for img in instances.get("images", [])}
    anns_by_id = {ann["id"]: ann for ann in instances.get("annotations", [])}
    categories_by_id = {cat["id"]: cat for cat in instances.get("categories", [])}
    return image_dir, refs, images_by_id, anns_by_id, categories_by_id


def _match_split(ref_split, target_split):
    if ref_split == target_split:
        return True
    if target_split == "test" and "test" in ref_split:
        return True
    return False


def _build_samples(data_root, split, language_filter="all"):
    image_dir, refs, images_by_id, anns_by_id, categories_by_id = _load_instances_and_refs(data_root)

    all_images = []
    all_ann_ids = []
    all_category_ids = []
    all_sentences = []
    language_counter = Counter()

    for ref in refs:
        ref_split = ref.get("split", "")
        if not _match_split(ref_split, split):
            continue

        img_id = ref.get("image_id")
        ann_id = ref.get("ann_id")
        category_id = ref.get("category_id")
        if img_id not in images_by_id or ann_id not in anns_by_id:
            continue

        file_name = images_by_id[img_id].get("file_name", "")
        if not file_name:
            continue
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            continue

        for sent in ref.get("sentences", []):
            sentence = sent.get("raw", "").strip()
            if not sentence:
                continue
            sentence_language = classify_text_language(sentence)
            language_counter[sentence_language] += 1
            if not text_matches_language_filter(sentence, language_filter):
                continue
            all_images.append(img_path)
            all_ann_ids.append(ann_id)
            all_category_ids.append(category_id)
            all_sentences.append(sentence)

    print(
        f"NWPU-refer split={split} lang={language_filter} loaded: {len(all_images)} samples "
        f"(available languages: {dict(language_counter)})"
    )
    return all_images, all_ann_ids, all_category_ids, all_sentences, anns_by_id, categories_by_id


def _ann_to_binary_mask(ann, height, width):
    segmentation = ann.get("segmentation")
    if segmentation:
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            if isinstance(segmentation.get("counts"), list):
                rle = mask_utils.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation
        else:
            rle = None

        if rle is not None:
            decoded = mask_utils.decode(rle)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            return decoded.astype(np.uint8)

    # Fallback to bbox when segmentation is unavailable.
    mask = np.zeros((height, width), dtype=np.uint8)
    bbox = ann.get("bbox", None)
    if bbox and len(bbox) == 4:
        x, y, w, h = bbox
        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))
        x2 = min(width, int(np.ceil(x + w)))
        y2 = min(height, int(np.ceil(y + h)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def _mark_phrase_tokens(mask, sentence_tokens, phrase_tokens, offset=1):
    if not phrase_tokens or len(phrase_tokens) > len(sentence_tokens):
        return False

    matched = False
    for start in range(len(sentence_tokens) - len(phrase_tokens) + 1):
        if sentence_tokens[start:start + len(phrase_tokens)] == phrase_tokens:
            for idx in range(start, start + len(phrase_tokens)):
                if idx + offset < len(mask):
                    mask[idx + offset] = 1
            matched = True
    return matched


class ReferDataset(data.Dataset):

    def __init__(
        self,
        args,
        image_transforms=None,
        target_transforms=None,
        split="train",
        eval_mode=False,
    ):
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.max_tokens = 22
        self.debug_diagnostics = getattr(args, "debug_diagnostics", False)
        self.debug_log_first_n = int(getattr(args, "debug_log_first_n", 3))
        self._debug_samples_logged = 0

        data_root = args.nwpu_data_root if getattr(args, "nwpu_data_root", "") else args.refer_data_root
        language_filter = getattr(args, "nwpu_lang", "all")
        (
            self.imgs,
            self.ann_ids,
            self.category_ids,
            self.sentences,
            self.anns_by_id,
            self.categories_by_id,
        ) = _build_samples(data_root, split, language_filter=language_filter)

        self.input_ids = []
        self.attention_masks = []
        self.target_masks = []
        self.position_masks = []
        self.sentences_raw = []
        self.model_sentences = []
        self.pp_phrase = []

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.eval_mode = eval_mode
        target_fallback_count = 0
        position_fallback_count = 0

        for sentence_raw, category_id in zip(self.sentences, self.category_ids):
            attention_mask = [0] * self.max_tokens
            padded_input_ids = [0] * self.max_tokens
            target_token_mask = [0] * self.max_tokens
            position_token_mask = [0] * self.max_tokens

            category_name = canonicalize_category_name(self.categories_by_id.get(category_id, {}).get("name", "object"))
            prompt_spec = build_prompt_spec(sentence_raw, category_name)
            model_sentence = prompt_spec.prompt
            sentence_tokens = self.tokenizer.tokenize(model_sentence)
            sentence_tokens = sentence_tokens[: self.max_tokens - 2]
            input_ids = self.tokenizer.encode(text=model_sentence, add_special_tokens=True)
            input_ids = input_ids[: self.max_tokens]

            padded_input_ids[: len(input_ids)] = input_ids
            attention_mask[: len(input_ids)] = [1] * len(input_ids)

            self.input_ids.append([torch.tensor(padded_input_ids).unsqueeze(0)])
            self.attention_masks.append([torch.tensor(attention_mask).unsqueeze(0)])

            self.sentences_raw.append(sentence_raw)
            self.model_sentences.append(model_sentence)

            matched_target = False
            for phrase in prompt_spec.target_phrases:
                phrase_tokens = self.tokenizer.tokenize(phrase)
                matched_target = _mark_phrase_tokens(target_token_mask, sentence_tokens, phrase_tokens) or matched_target
            target_tensor = torch.tensor(target_token_mask).unsqueeze(0)
            if not matched_target or torch.sum(target_tensor) == 0:
                target_tensor = self.attention_masks[-1][0]
                target_fallback_count += 1
            self.target_masks.append([target_tensor])

            matched_position = False
            for phrase in prompt_spec.position_phrases:
                phrase_tokens = self.tokenizer.tokenize(phrase)
                matched_position = _mark_phrase_tokens(position_token_mask, sentence_tokens, phrase_tokens) or matched_position
            position_tensor = torch.tensor(position_token_mask).unsqueeze(0)
            if not matched_position or torch.sum(position_tensor) == 0:
                position_tensor = self.attention_masks[-1][0]
                position_fallback_count += 1
                self.pp_phrase.append([])
            else:
                self.pp_phrase.append(list(prompt_spec.position_phrases))
            self.position_masks.append([position_tensor])

        if len(self.sentences) > 0:
            target_ratio = target_fallback_count / float(len(self.sentences))
            position_ratio = position_fallback_count / float(len(self.sentences))
            print(
                f"NWPU target-mask fallback: {target_fallback_count}/{len(self.sentences)} "
                f"({target_ratio:.2%})"
            )
            print(
                f"NWPU position-mask fallback: {position_fallback_count}/{len(self.sentences)} "
                f"({position_ratio:.2%})"
            )

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        ann = self.anns_by_id[self.ann_ids[index]]
        ref_mask = _ann_to_binary_mask(ann, height, width) > 0
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        save_prefix = f"{index}_{self.sentences_raw[index]}"

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)

        if self.debug_diagnostics and self._debug_samples_logged < self.debug_log_first_n:
            raw_fg = int(np.count_nonzero(ref_mask))
            raw_total = int(ref_mask.size)
            transformed_fg = int(torch.count_nonzero(target).item())
            transformed_total = int(target.numel())
            print(
                f"[Debug][Dataset][{self.split}] idx={index} ann_id={self.ann_ids[index]} "
                f"raw_fg={raw_fg}/{raw_total} ({raw_fg / max(raw_total, 1):.4%}) "
                f"transformed_fg={transformed_fg}/{transformed_total} "
                f"({transformed_fg / max(transformed_total, 1):.4%}) "
                f"sentence={self.sentences_raw[index]!r}"
            )
            self._debug_samples_logged += 1

        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]
        target_mask = self.target_masks[index][choice_sent]
        position_mask = self.position_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask, target_mask, position_mask, save_prefix
