import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import _pickle as pickle
import sys
import os
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils import data
import collections
import numpy as np
from tqdm import tqdm

from model.model import FWUNet
from common.data_utils import *
from datasets.data_generator import RNADataset, BucketBatchSampler, collate_fn
from common.postprocess import *
from common.loss_utils import AsymmetricLoss

from config.config import get_config

perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2],
           [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]

def model_eval_all_test(contact_net, test_generator):
    device = next(contact_net.parameters()).device
    contact_net.eval()

    result_no_train = []
    result_nc = []

    test_pbar = tqdm(test_generator, desc="Evaluating", leave=False, unit="batch")

    for batch in test_pbar:
        contact, data_fcn_2, combined_pairwise, data_length, seq_raw, name, _ = batch

        contacts_batch = contact.float().to(device)
        data_fcn_2 = data_fcn_2.float().to(device)
        combined_pairwise = combined_pairwise.float().to(device)
        data_length = data_length.float().to(device)

        batch_size = contacts_batch.shape[0]
        num_base_types = data_fcn_2.shape[-1]

        nc_map = torch.zeros_like(contacts_batch, dtype=torch.bool, device=device)

        for b in range(batch_size):
            data_len = int(data_length[b].item())
            data_seq = data_fcn_2[b]

            for bi, bj in perm_nc:
                if bi >= num_base_types or bj >= num_base_types:
                    continue
                mat = torch.matmul(
                    data_seq[:data_len, bi].view(-1, 1),
                    data_seq[:data_len, bj].view(1, -1),
                )
                nc_map[b, :data_len, :data_len] |= (mat > 0)

        nc_map_nc = nc_map.float() * contacts_batch

        with torch.no_grad():
            pred_contacts = contact_net(combined_pairwise)

        if isinstance(pred_contacts, list):
            pred_contacts = pred_contacts[0]

        u_no_train = postprocess(
            pred_contacts,
            data_fcn_2,
            0.01, 0.1, 100, 1.6, True, 1.5,
        )
        map_no_train = (u_no_train > 0.5).float()

        for i in range(batch_size):
            pred_cpu = map_no_train[i].cpu()
            target_cpu = contacts_batch[i].cpu()

            result_no_train.append(evaluate_exact_new(pred_cpu, target_cpu))

            if nc_map_nc[i].sum() > 0:
                nc_target = nc_map_nc[i].cpu().float()
                nc_pred = (
                        (nc_map[i].float().cpu() * u_no_train[i].cpu()) > 0.5
                ).float()
                result_nc.append(evaluate_exact_new(nc_pred, nc_target))

    if len(result_no_train) == 0:
        return 0.0, 0.0, 0.0, []

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)

    return (
        float(np.mean(nt_exact_f1)),
        float(np.mean(nt_exact_p)),
        float(np.mean(nt_exact_r)),
        result_nc,
    )


def main():
    seed_torch()

    cfg = get_config()

    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    weight_path = cfg.get("weight_path", None)
    if weight_path is None:
        weight_path = os.path.join(cfg["save_dir"], "tufold_latest.pt")

    if not os.path.exists(weight_path):
        print(f"Warning: Weight file not found at {weight_path}")
        print("Please check your '--weight_path' argument or hardcode the path.")
        return

    print("===== Loading Datasets =====")
    dataset_test = RNADataset(
        cfg["test_root"],
        dataset_name=cfg["test_files"],
        cache_dir=cfg["cache_test"],
        is_train=False,
        is_test=True,
        upsample=False
    )

    print(f"Test Set: {cfg['test_files']} | Size: {len(dataset_test)}")

    batch_size = 1
    sampler_test = BucketBatchSampler(dataset_test, batch_size=batch_size, shuffle=False)

    loader_test = data.DataLoader(
        dataset_test,
        batch_sampler=sampler_test,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    contact_net = FWUNet(
        img_ch=cfg.get("img_ch", 17),
        output_ch=cfg.get("output_ch", 1),
        wave=cfg.get("wave", "haar")
    ).to(device)

    print(f"Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)

    if 'module.' in list(checkpoint.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        contact_net.load_state_dict(new_state_dict)
    else:
        contact_net.load_state_dict(checkpoint)

    print("Model loaded successfully.")

    avg_f1, avg_p, avg_r, result_nc = model_eval_all_test(contact_net, loader_test)

    print("\n" + "=" * 35)
    print(f"FINAL TEST RESULTS ({cfg['test_files']})")
    print(f"F1 Score : {avg_f1:.4f}")
    print(f"Precision: {avg_p:.4f}")
    print(f"Recall   : {avg_r:.4f}")

    if len(result_nc) > 0:
        nc_p, nc_r, nc_f1 = zip(*result_nc)
        print(f"Non-Canonical F1: {np.average(nc_f1):.4f} (on {len(result_nc)} samples)")
    print("=" * 35)

if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple(
        'RNA_SS_data', 'data_fcn_2 seq_raw length name contact'
    )
    main()
