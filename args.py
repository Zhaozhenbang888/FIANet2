import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='FIANet training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='./bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
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
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--num_tmem', default=1, type=int, help='number of tmem layers')  # 1 for RefSegRS, 3 for RRSIS-D
    parser.add_argument('--grl_hidden_dim', default=256, type=int, help='hidden dim of GRL graph reasoning module')
    parser.add_argument('--grl_num_nodes', default=64, type=int, help='number of visual graph nodes in GRL')
    parser.add_argument('--grl_num_steps', default=2, type=int, help='number of GAT+GCN reasoning steps in GRL')
    parser.add_argument('--grl_drop', default=0.1, type=float, help='dropout rate for GRL graph attention')
    parser.add_argument(
        '--grl_mode',
        default='full',
        choices=['full', 'no_parser', 'off'],
        help='GRL ablation mode: full uses the fine-grained parser, no_parser keeps GRL but disables structured parsing, off bypasses GRL entirely.',
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
