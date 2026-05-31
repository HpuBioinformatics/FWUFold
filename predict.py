import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils import data
import collections
import warnings

from model.model import FWUNet
from common.data_utils import seed_torch
from datasets.data_generator import collate_fn,FASTARNADataset,BucketBatchSampler
from common.postprocess import postprocess

from config.config import get_config

def save_bpseq(seq, pairs, filepath, name):
    with open(filepath, 'w') as f:
        f.write(f"# {name}\n")
        for i in range(len(seq)):
            pair_idx = 0
            for p in pairs:
                if p[0] == i:
                    pair_idx = p[1] + 1
                elif p[1] == i:
                    pair_idx = p[0] + 1
            f.write(f"{i + 1} {seq[i]} {pair_idx}\n")


def save_ct(seq, pairs, filepath, name):
    with open(filepath, 'w') as f:
        f.write(f"{len(seq)} {name}\n")
        for i in range(len(seq)):
            pair_idx = 0
            for p in pairs:
                if p[0] == i:
                    pair_idx = p[1] + 1
                elif p[1] == i:
                    pair_idx = p[0] + 1
            prev_idx = i if i > 0 else 0
            next_idx = i + 2 if i + 1 < len(seq) else 0
            f.write(f"{i + 1} {seq[i]} {prev_idx} {next_idx} {pair_idx} {i + 1}\n")

def predict_all(contact_net, test_generator, cfg):
    device = next(contact_net.parameters()).device
    contact_net.eval()

    out_dir = cfg.get('output_dir', './predictions')
    os.makedirs(out_dir, exist_ok=True)

    test_pbar = tqdm(test_generator, desc="Predicting", leave=False, unit="batch")

    for batch in test_pbar:
        data_fcn_2 = batch[1].float().to(device)
        combined_pairwise = batch[2].float().to(device)
        data_length = batch[3]
        seq_raw_list = batch[4]
        name_list = batch[5]

        with torch.no_grad():
            pred_contacts = contact_net(combined_pairwise)

        if isinstance(pred_contacts, list):
            pred_contacts = pred_contacts[0]

        u_no_train = postprocess(
            pred_contacts, data_fcn_2, 0.01, 0.1, 100, 1.6, True, 1.5
        )

        batch_size_current = combined_pairwise.shape[0]

        for b in range(batch_size_current):
            L = int(data_length[b].item())
            seq = seq_raw_list[b]
            seq_name = name_list[b]

            safe_seq_name = str(seq_name).replace('/', '_').replace('\\', '_').replace('|', '_').replace(' ', '_')

            pred_map = (u_no_train[b, :L, :L] > 0.5).cpu().numpy()

            pairs = []
            for i in range(L):
                for j in range(i + 1, L):
                    if pred_map[i, j] == 1:
                        pairs.append((i, j))

            bpseq_path = os.path.join(out_dir, f"{safe_seq_name}.bpseq")
            save_bpseq(seq, pairs, bpseq_path, seq_name)  # 文件内部的 header 依然保留原名

            ct_path = os.path.join(out_dir, f"{safe_seq_name}.ct")
            save_ct(seq, pairs, ct_path, seq_name)

def main():
    seed_torch()
    cfg = get_config()

    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print(f"Using device: {device}")

    weight_path = cfg.get("weight_path", None)
    if weight_path is None:
        weight_path = os.path.join(cfg.get("save_dir", "./"), "fwufold_latest.pt")

    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        return

    print("\n===== Preparing Input Data =====")

    f_file = cfg.get("fasta_file")
    r_seq = cfg.get("raw_seq")

    if f_file is None and r_seq is None:
        print("Error: Please provide the FASTA file path or plain text sequence via the '--input' argument.")
        return

    if f_file:
        print(f"Loading FASTA file: {f_file}")
    else:
        print(f"Processing raw sequence input...")

    dataset_predict = FASTARNADataset(
        fasta_file=f_file,
        raw_seq=r_seq,
        seq_name=cfg.get("seq_name", "Test_seq")
    )

    print(f"Found {len(dataset_predict)} sequence(s) for prediction.")

    batch_size = cfg.get("batch_size", 4)
    sampler_predict = BucketBatchSampler(dataset_predict, batch_size=batch_size, shuffle=False)

    loader_predict = data.DataLoader(
        dataset_predict,
        batch_sampler=sampler_predict,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    print("\n===== Loading Model =====")
    contact_net = FWUNet(
        img_ch=cfg.get("img_ch", 17),
        output_ch=cfg.get("output_ch", 1),
        base_len=cfg.get("base_len", 640),
        wave=cfg.get("wave", "haar")
    ).to(device)

    checkpoint = torch.load(weight_path, map_location=device)
    if 'module.' in list(checkpoint.keys())[0]:
        contact_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    else:
        contact_net.load_state_dict(checkpoint)

    print("Model loaded successfully.")

    print("\n===== Starting Prediction =====")
    predict_all(contact_net, loader_predict, cfg)

    print("\n===== Prediction Finished! =====")
    print(f"Results are saved in: {os.path.abspath(cfg.get('output_dir', './predictions'))}")

if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'data_fcn_2 seq_raw length name contact')
    main()