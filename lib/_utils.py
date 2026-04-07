import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


def load_weights(model, load_path):
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    print('load weights from {}'.format(load_path))
    return model


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        self.text_encoder_zh = None
        self.text_route_mode = getattr(args, "text_route_mode", "single")
        if self.text_route_mode == "dual":
            zh_ckpt = getattr(args, "ck_bert_zh", "") or args.ck_bert
            try:
                self.text_encoder_zh = BertModel.from_pretrained(zh_ckpt)
                self.text_encoder_zh.pooler = None
            except Exception as exc:
                print(
                    f"[Warning][TextEncoder] Failed to load Chinese text encoder '{zh_ckpt}': {exc}. "
                    "Fallback to single English text encoder."
                )
                self.text_encoder_zh = None

    def _encode_text(self, text, attention_mask, text_lang_ids):
        if self.text_encoder_zh is None or text_lang_ids is None:
            return self.text_encoder(text, attention_mask=attention_mask)[0]

        if text_lang_ids.dim() > 1:
            text_lang_ids = text_lang_ids.view(text_lang_ids.size(0), -1)[:, 0]

        text_lang_ids = text_lang_ids.to(text.device).long()
        batch_size = text.size(0)
        hidden_states = None

        english_mask = text_lang_ids == 0
        chinese_mask = text_lang_ids == 1
        other_mask = ~(english_mask | chinese_mask)

        if torch.any(english_mask):
            english_states = self.text_encoder(
                text[english_mask], attention_mask=attention_mask[english_mask]
            )[0]
            hidden_states = english_states.new_zeros((batch_size, english_states.size(1), english_states.size(2)))
            hidden_states[english_mask] = english_states

        if torch.any(chinese_mask):
            chinese_states = self.text_encoder_zh(
                text[chinese_mask], attention_mask=attention_mask[chinese_mask]
            )[0]
            if hidden_states is None:
                hidden_states = chinese_states.new_zeros((batch_size, chinese_states.size(1), chinese_states.size(2)))
            hidden_states[chinese_mask] = chinese_states

        if torch.any(other_mask):
            other_states = self.text_encoder(
                text[other_mask], attention_mask=attention_mask[other_mask]
            )[0]
            if hidden_states is None:
                hidden_states = other_states.new_zeros((batch_size, other_states.size(1), other_states.size(2)))
            hidden_states[other_mask] = other_states

        return hidden_states

    def forward(self, x, text, l_mask, t_mask, p_mask, text_lang_ids=None):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self._encode_text(text, l_mask, text_lang_ids)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l)
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        t_feats = self._encode_text(text, t_mask, text_lang_ids)
        t_feats = t_feats.permute(0, 2, 1)  # (B, 768, N_l)
        t_mask = t_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        p_feats = self._encode_text(text, p_mask, text_lang_ids)
        p_feats = p_feats.permute(0, 2, 1)  # (B, 768, N_l)
        p_mask = p_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        # Keep raw token ids for GRL and let GRL route parser behavior per-sample.
        text_ids_for_grl = text

        ##########################
        features = self.backbone(
            x,
            l_feats,
            l_mask,
            t_feats,
            t_mask,
            p_feats,
            p_mask,
            text_ids=text_ids_for_grl,
            text_lang_ids=text_lang_ids,
        )
        x_c1, x_c2, x_c3, x_c4  = features   # e.g. x_c1:[B, 128, 120, 120], x_c2:[B, 256, 60, 60], x_c3:[B, 512, 30, 30], x_c4:[B, 1024, 15, 15]
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        return x


class LAVTOne(_LAVTOneSimpleDecode):  #change
    pass
