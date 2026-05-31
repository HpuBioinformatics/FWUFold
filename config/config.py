import os
import json
import argparse

BASE_DIR = "./data"


DATA_ROOT_MAP = {
    "RNAStrAlign": {
        "train": f"{BASE_DIR}/RNAStrAlign/train",
        "val": f"{BASE_DIR}/RNAStrAlign/val",
        "test": f"{BASE_DIR}/RNAStrAlign/test"
    },
    "bpRNA": {
        "train": f"{BASE_DIR}/bpRNA/TR0",
        "val": f"{BASE_DIR}/bpRNA/VL0",
        "test": f"{BASE_DIR}/bpRNA/TS0"
    },
    "ArchiveII": {
        "test": f"{BASE_DIR}/ArchiveII"
    },
    "mutate": {
        "train": f"{BASE_DIR}/allbpnew_mutate"
    },
    "PDB": {
        "train": f"{BASE_DIR}/PDB/TR1",
        "val": f"{BASE_DIR}/PDB/VL1",
    },
    "TS1": {
        "val": f"{BASE_DIR}/PDB/TS1",
        "test": f"{BASE_DIR}/PDB/TS1"
    },
    "TS2": {
        "val": f"{BASE_DIR}/PDB/TS2",
        "test": f"{BASE_DIR}/PDB/TS2"
    },
    "TS3": {
        "val": f"{BASE_DIR}/PDB/TS3",
        "test": f"{BASE_DIR}/PDB/TS3"
    },
    "TS_hard": {
        "val": f"{BASE_DIR}/PDB/TS_hard",
        "test": f"{BASE_DIR}/PDB/TS_hard"
    },

    "TS123": {
        "val": [
            f"{BASE_DIR}/PDB/TS1",
            f"{BASE_DIR}/PDB/TS2",
            f"{BASE_DIR}/PDB/TS3"
        ],
        "test": [
            f"{BASE_DIR}/PDB/TS1",
            f"{BASE_DIR}/PDB/TS2",
            f"{BASE_DIR}/PDB/TS3"
        ]
    },
    "bpRNA-new": {
        "train": [
            f"{BASE_DIR}/bpRNA/TR0",
            f"{BASE_DIR}/allbpnew_mutate"
        ],
        "val": f"{BASE_DIR}/bpRNA-new",
        "test": f"{BASE_DIR}/bpRNA-new"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="RNA Secondary Structure Prediction Training & Inference")

    parser.add_argument('--test', action='store_true', help='Skip training to test directly.')
    parser.add_argument('--nc', action='store_true', help='Whether predict non-canonical pairs.')

    # 训练集
    parser.add_argument('--train_files', type=str, nargs='+', default=['RNAStrAlign', 'bpRNA'],
                        choices=['RNAStrAlign', 'bpRNA', 'ArchiveII', 'PDB', 'mutate', 'bpRNA-new'],
                        help='Training dataset name list.')

    # 验证集
    parser.add_argument('--val_files', type=str, nargs='+', default=['RNAStrAlign', 'bpRNA'],
                        choices=['RNAStrAlign', 'bpRNA', 'ArchiveII', 'PDB', 'bpRNA-new', 'TS1', 'TS2', 'TS3',
                                 'TS_hard', 'TS123'],
                        help='Validation dataset name list.')

    # 测试集
    parser.add_argument('--test_files', type=str, default='ArchiveII',
                        choices=['RNAStrAlign', 'bpRNA', 'ArchiveII', 'PDB', 'mutate', 'bpRNA-new', 'TS1', 'TS2', 'TS3',
                                 'TS_hard', 'TS123'],
                        help='Test dataset name')

    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save weights. Auto-generated if not specified.')

    parser.add_argument('--weight_path', type=str, default=None, help='Path to the trained model weights for testing.')

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--input', type=str, default=None,
                        help='Input for prediction: FASTA file path OR raw RNA sequence string.')

    parser.add_argument('--seq_name', type=str, default='Test_seq',
                        help='Default name for the sequence if providing a raw string via --input.')

    parser.add_argument('--output_dir', type=str, default='./predictions_new',
                        help='Directory to save predicted .bpseq and .ct files.')

    return parser.parse_args()

def get_config():
    args = parse_args()

    json_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        json_cfg = json.load(f)

    train_roots = []
    for ds_name in args.train_files:
        if "train" in DATA_ROOT_MAP[ds_name]:
            train_roots.append(DATA_ROOT_MAP[ds_name]["train"])

    val_roots = []
    for ds_name in args.val_files:
        if "val" in DATA_ROOT_MAP[ds_name]:
            val_roots.append(DATA_ROOT_MAP[ds_name]["val"])

    test_root = [DATA_ROOT_MAP[args.test_files].get("test", "")]

    if args.save_dir is None:
        dataset_prefix = "_".join(args.train_files)
        save_dir = f"./weight/{dataset_prefix}"
    else:
        save_dir = args.save_dir

    cache_train = f"./cache/{'_'.join(args.train_files)}/train/"
    cache_val = f"./cache/{'_'.join(args.val_files)}/val/"
    cache_test = f"./cache/{args.test_files}/test/"

    fasta_file = None
    raw_seq = None

    if args.input is not None:
        if os.path.isfile(args.input) or args.input.lower().endswith(('.fasta', '.fa', '.txt', '.seq')):
            fasta_file = args.input
        else:
            raw_seq = args.input

    cfg = {
        **json_cfg.get("model", {}),
        **json_cfg.get("train", {}),
        **json_cfg.get("loss", {}),

        "gpu_id": args.gpu_id,
        "test_mode": args.test,
        "predict_nc": args.nc,

        "train_files": args.train_files,
        "val_files": args.val_files,
        "test_files": args.test_files,

        "save_dir": save_dir,
        "train_roots": train_roots,
        "val_roots": val_roots,
        "test_root": test_root,
        "cache_train": cache_train,
        "cache_val": cache_val,
        "cache_test": cache_test,

        "weight_path": args.weight_path,

        "fasta_file": fasta_file,
        "raw_seq": raw_seq,
        "seq_name": args.seq_name,
        "output_dir": args.output_dir
    }

    return cfg



if __name__ == "__main__":
    config = get_config()
    print("=" * 40)
    print("Generated Configuration Info:")
    for k, v in config.items():
        if k in ["fasta_file", "raw_seq", "seq_name", "output_dir", "test_mode"]:
            print(f"{k}: {v}")
    print("=" * 40)