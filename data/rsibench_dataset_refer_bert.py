import os
import re

import cv2
import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from nltk.tokenize import word_tokenize

from bert.tokenization_bert import BertTokenizer
from lib.rs_vocabulary import RISBENCH_ENTITIES


def _resolve_rsibench_paths(data_root, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split={split} for RISBench_dataset")

    candidate_roots = [
        data_root,
        os.path.join(data_root, "RISBench_dataset"),
    ]

    for root in candidate_roots:
        image_dir = os.path.join(root, "img_rgb")
        mask_dir = os.path.join(root, "mask")
        split_file = os.path.join(root, f"output_phrase_{split}.txt")
        if os.path.isdir(image_dir) and os.path.isdir(mask_dir) and os.path.isfile(split_file):
            return image_dir, mask_dir, split_file

    raise FileNotFoundError(
        "Cannot resolve RISBench_dataset paths. "
        f"Expected img_rgb/, mask/, and output_phrase_{split}.txt under {data_root}"
    )


def _build_samples(data_root, split):
    image_dir, mask_dir, split_file = _resolve_rsibench_paths(data_root, split)

    all_images = []
    all_masks = []
    all_sentences = []

    with open(split_file, "r", encoding="utf-8") as rf:
        for line_idx, line in enumerate(rf, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid line format in {split_file}:{line_idx}: {line}")

            sample_name, sentence = parts[0], parts[1].strip()
            img_path = os.path.join(image_dir, sample_name)
            mask_path = os.path.join(mask_dir, sample_name)

            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found for {split_file}:{line_idx}: {img_path}")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Mask not found for {split_file}:{line_idx}: {mask_path}")

            all_images.append(img_path)
            all_masks.append(mask_path)
            all_sentences.append(sentence)

    print(f"RSIBench_dataset split={split} loaded: {len(all_images)} samples")
    return all_images, all_masks, all_sentences


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

        data_root = args.rsibench_data_root if getattr(args, "rsibench_data_root", "") else args.refer_data_root
        self.imgs, self.labels, self.sentences = _build_samples(data_root, split)

        self.input_ids = []
        self.attention_masks = []
        self.target_masks = []
        self.position_masks = []
        self.text_language_ids = []
        self.sentences_raw = []
        self.pp_phrase = []

        self.target_cls = set(RISBENCH_ENTITIES)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.eval_mode = eval_mode

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

            for cls_name in self.target_cls:
                if re.findall(cls_name, sentence_raw):
                    tokenized_cls = word_tokenize(cls_name)
                    cls_len = len(tokenized_cls)
                    cls_start = 0
                    for idx, token in enumerate(tokenized_sentence):
                        if re.findall(tokenized_cls[0], token):
                            cls_start = idx
                            break
                    target_token_mask[cls_start + 1 : cls_start + cls_len + 1] = [1] * cls_len

            self.target_masks.append([torch.tensor(target_token_mask).unsqueeze(0)])

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
            self.text_language_ids.append([torch.tensor([0], dtype=torch.long)])

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        mask_path = self.labels[index]

        img = Image.open(img_path).convert("RGB")
        label_mask = cv2.imread(mask_path, 2)
        if label_mask is None:
            raise FileNotFoundError(f"Failed to read mask file: {mask_path}")

        ref_mask = np.array(label_mask) > 50
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
        text_language_id = self.text_language_ids[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask, target_mask, position_mask, text_language_id, save_prefix
