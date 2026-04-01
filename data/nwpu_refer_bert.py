import os
import re
import json
import pickle

import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from nltk.tokenize import word_tokenize
from pycocotools import mask as mask_utils

from bert.tokenization_bert import BertTokenizer
from lib.rs_vocabulary import NWPU_ENTITIES


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
    return image_dir, refs, images_by_id, anns_by_id


def _match_split(ref_split, target_split):
    if ref_split == target_split:
        return True
    if target_split == "test" and "test" in ref_split:
        return True
    return False


def _build_samples(data_root, split):
    image_dir, refs, images_by_id, anns_by_id = _load_instances_and_refs(data_root)

    all_images = []
    all_ann_ids = []
    all_sentences = []

    for ref in refs:
        ref_split = ref.get("split", "")
        if not _match_split(ref_split, split):
            continue

        img_id = ref.get("image_id")
        ann_id = ref.get("ann_id")
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
            all_images.append(img_path)
            all_ann_ids.append(ann_id)
            all_sentences.append(sentence)

    print(f"NWPU-refer split={split} loaded: {len(all_images)} samples")
    return all_images, all_ann_ids, all_sentences, anns_by_id


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

        data_root = args.nwpu_data_root if getattr(args, "nwpu_data_root", "") else args.refer_data_root
        self.imgs, self.ann_ids, self.sentences, self.anns_by_id = _build_samples(data_root, split)

        self.input_ids = []
        self.attention_masks = []
        self.target_masks = []
        self.position_masks = []
        self.sentences_raw = []
        self.pp_phrase = []

        self.target_cls = set(NWPU_ENTITIES)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.eval_mode = eval_mode
        target_fallback_count = 0

        for sentence_raw in self.sentences:
            attention_mask = [0] * self.max_tokens
            padded_input_ids = [0] * self.max_tokens
            target_token_mask = [0] * self.max_tokens
            position_token_mask = [0] * self.max_tokens

            input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
            input_ids = input_ids[: self.max_tokens]

            padded_input_ids[: len(input_ids)] = input_ids
            attention_mask[: len(input_ids)] = [1] * len(input_ids)

            self.input_ids.append([torch.tensor(padded_input_ids).unsqueeze(0)])
            self.attention_masks.append([torch.tensor(attention_mask).unsqueeze(0)])

            self.sentences_raw.append(sentence_raw)
            tokenized_sentence = word_tokenize(sentence_raw)
            tokenized_sentence_lower = [token.lower() for token in tokenized_sentence]
            sentence_lower = sentence_raw.lower()

            for cls_name in self.target_cls:
                cls_name_lower = cls_name.lower()
                if re.search(re.escape(cls_name_lower), sentence_lower):
                    tokenized_cls = word_tokenize(cls_name)
                    cls_len = len(tokenized_cls)
                    tokenized_cls_lower = [token.lower() for token in tokenized_cls]
                    cls_start = 0
                    for idx, token in enumerate(tokenized_sentence):
                        if tokenized_cls_lower and tokenized_cls_lower[0] == tokenized_sentence_lower[idx]:
                            cls_start = idx
                            break
                    target_token_mask[cls_start + 1 : cls_start + cls_len + 1] = [1] * cls_len
            target_tensor = torch.tensor(target_token_mask).unsqueeze(0)
            if torch.sum(target_tensor) == 0:
                target_tensor = self.attention_masks[-1][0]
                target_fallback_count += 1
            self.target_masks.append([target_tensor])

            grammar = r"""
            PP: {<IN><DT>?<JJ.*>?<NN>}
                {<IN><DT>?<JJ.*>?<JJ>}
                {<IN><DT>?<JJ.*><VBD>}
            """
            chunkr = nltk.RegexpParser(grammar)
            tree = chunkr.parse(nltk.pos_tag(tokenized_sentence))
            pp_phrases = []
            for subtree in tree.subtrees():
                if subtree.label() == "PP":
                    pp_phrases.append(" ".join(word for word, _ in subtree.leaves()))

            valid_pp = [phrase for phrase in pp_phrases if not re.findall("of", phrase)]
            if valid_pp:
                for pp in valid_pp:
                    tokenized_pos = word_tokenize(pp)
                    pos_len = len(tokenized_pos)
                    pos_start = 0
                    for idx, token in enumerate(tokenized_sentence):
                        if tokenized_pos[0] == token:
                            pos_start = idx
                            break
                    position_token_mask[pos_start + 1 : pos_start + pos_len + 1] = [1] * pos_len

            self.pp_phrase.append(valid_pp)
            position_tensor = torch.tensor(position_token_mask).unsqueeze(0)
            if torch.sum(position_tensor) == 0:
                position_tensor = self.attention_masks[-1][0]
            self.position_masks.append([position_tensor])

        if len(self.sentences) > 0:
            fallback_ratio = target_fallback_count / float(len(self.sentences))
            print(
                f"NWPU target-mask fallback: {target_fallback_count}/{len(self.sentences)} "
                f"({fallback_ratio:.2%})"
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

        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]
        target_mask = self.target_masks[index][choice_sent]
        position_mask = self.position_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask, target_mask, position_mask, save_prefix
