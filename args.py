import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='FIANet training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='./bert-base-uncased', help='BERT tokenizer')
    parser.add_argument(
        '--bert_tokenizer_zh',
        default='hfl/chinese-roberta-wwm-ext',
        help='Chinese tokenizer used when text_route_mode=dual; fallback to bert_tokenizer if loading fails',
    )
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument(
        '--ck_bert_zh',
        default='hfl/chinese-roberta-wwm-ext',
        help='Chinese BERT weights used when text_route_mode=dual; fallback to ck_bert if loading fails',
    )
    parser.add_argument(
        '--text_route_mode',
        default='dual',
        choices=['single', 'dual'],
        help='single uses one English text encoder; dual routes English to English encoder and Chinese to Chinese encoder.',
    )
    parser.add_argument('--dataset', default='rrsisd',
                        help='dataset name: refsegrs, rrsisd, nwpu-refer, rsibench_dataset')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int,default=0,help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=5e-5, type=float, help='the initial learning rate')   # 5e-5 for RefSegRS, 3e-5 for RRSIS-D
    parser.add_argument('--lr_warmup_steps', default=1000, type=int,
                        help='linear warmup steps for learning rate; helps stabilize early training')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='FIANet', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='./pretrained_weights/swin_base_patch4_window12_384_22k.pth',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='C:/Dataset/refer_seg/RefSegRS/', help='REFER dataset root directory')
    parser.add_argument('--nwpu_data_root', default='',
                        help='NWPU-refer dataset root; fallback to refer_data_root when empty')
    parser.add_argument(
        '--nwpu_lang',
        default='all',
        choices=['all', 'english', 'chinese'],
        help='language subset for NWPU-refer: all keeps every sentence, english keeps English-only sentences, chinese keeps Chinese-only sentences.',
    )
    parser.add_argument('--rsibench_data_root', default='',
                        help='RSIBench_dataset root; fallback to refer_data_root when empty')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the datasset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--grad_clip_norm', default=1.0, type=float,
                        help='max grad norm for gradient clipping; set <=0 to disable')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument(
        '--cudnn_deterministic',
        action='store_true',
        help='force deterministic cuDNN algorithms for reproducibility; may reduce available kernels on some GPUs',
    )
    parser.add_argument(
        '--disable_cudnn',
        action='store_true',
        help='disable cuDNN and use native PyTorch CUDA kernels when cuDNN algorithm selection fails',
    )
    parser.add_argument('--num_tmem', default=1, type=int, help='number of tmem layers')  # 1 for RefSegRS, 3 for RRSIS-D
    parser.add_argument('--grl_hidden_dim', default=256, type=int, help='hidden dim of GRL graph reasoning module')
    parser.add_argument('--grl_num_nodes', default=64, type=int, help='number of visual graph nodes in GRL')
    parser.add_argument('--grl_num_steps', default=2, type=int, help='number of GAT+GCN reasoning steps in GRL')
    parser.add_argument('--grl_drop', default=0.1, type=float, help='dropout rate for GRL graph attention')
    parser.add_argument(
        '--grl_residual_scale',
        default=0.2,
        type=float,
        help='global scale for GRL residual injection; lower values make GRL updates more conservative',
    )
    parser.add_argument(
        '--grl_residual_clip',
        default=1.0,
        type=float,
        help='clip value for GRL residual delta before injecting back to visual features; set <=0 to disable',
    )
    parser.add_argument(
        '--grl_mode',
        default='full',
        choices=['full', 'no_parser', 'off'],
        help='GRL ablation mode: full uses the fine-grained parser, no_parser keeps GRL but disables structured parsing, off bypasses GRL entirely.',
    )
    parser.add_argument(
        '--debug_diagnostics',
        action='store_true',
        help='print extra dataset/train/eval/backbone diagnostics for debugging zero-IoU runs',
    )
    parser.add_argument(
        '--debug_log_first_n',
        default=3,
        type=int,
        help='how many per-sample/per-stage debug lines to print before switching to summary-only logs',
    )
    parser.add_argument(
        '--loss_dice_weight',
        default=0.1,
        type=float,
        help='blend factor for Dice loss in total segmentation loss',
    )
    parser.add_argument(
        '--loss_fg_max_weight',
        default=8.0,
        type=float,
        help='upper bound of dynamic foreground class weight in cross entropy',
    )
    parser.add_argument(
        '--loss_fg_weight_exponent',
        default=0.35,
        type=float,
        help='exponent for dynamic foreground class weighting; smaller is less aggressive',
    )
    parser.add_argument(
        '--loss_ce_weight_mode',
        default='dynamic',
        choices=['dynamic', 'none', 'fixed'],
        help='cross entropy class-weight mode: dynamic uses per-sample balance, none disables class reweighting, fixed uses loss_fixed_fg_weight.',
    )
    parser.add_argument(
        '--loss_fixed_fg_weight',
        default=1.0,
        type=float,
        help='foreground class weight when loss_ce_weight_mode=fixed; background weight stays 1.0',
    )
    parser.add_argument(
        '--decoder_fg_prior',
        default=-1.0,
        type=float,
        help='if in (0,1), initialize decoder foreground logit bias from this prior to reduce early foreground collapse',
    )
    parser.add_argument(
        '--loss_ce_type',
        default='ce',
        choices=['ce', 'focal'],
        help='cross entropy variant: ce uses standard CE, focal uses focal CE to reduce easy-background dominance',
    )
    parser.add_argument(
        '--loss_focal_gamma',
        default=2.0,
        type=float,
        help='gamma for focal CE when loss_ce_type=focal',
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
