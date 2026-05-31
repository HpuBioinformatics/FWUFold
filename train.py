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

def get_wsd_scheduler(optimizer: torch.optim.Optimizer,
                      total_steps: int,
                      num_warmup_steps: int,
                      num_stable_steps: int,
                      min_lr_ratio: float = 0.01):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-4, float(current_step) / float(max(1, num_warmup_steps)))
        elif current_step < (num_warmup_steps + num_stable_steps):
            return 1.0
        else:
            decay_steps = total_steps - num_warmup_steps - num_stable_steps
            current_decay_step = current_step - num_warmup_steps - num_stable_steps
            progress = float(current_decay_step) / float(max(1, decay_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(contact_net, train_loader, cfg, val_loaders=None):
    device = next(contact_net.parameters()).device
    epoches_first = cfg["epochs"]

    criterion_asl = AsymmetricLoss(
        gamma_neg=cfg["asl_gamma_neg"],
        gamma_pos=cfg["asl_gamma_pos"],
        clip=cfg["asl_clip"],
        pos_weight=cfg["asl_pos_weight"],
        reduction='none'
    ).to(device)
    print(f"Loss: AsymmetricLoss "
          f"(γ-={cfg['asl_gamma_neg']}, γ+={cfg['asl_gamma_pos']}, "
          f"clip={cfg['asl_clip']}, pos_w={cfg['asl_pos_weight']})")

    optimizer = optim.AdamW(contact_net.parameters(), lr=cfg["lr"])

    steps_per_epoch = len(train_loader)
    total_steps = epoches_first * steps_per_epoch

    num_warmup_steps = int(0.10 * total_steps)
    num_stable_steps = int(0.70 * total_steps)

    scheduler = get_wsd_scheduler(
        optimizer,
        total_steps=total_steps,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        min_lr_ratio=0.01
    )

    save_path = cfg["save_dir"]
    os.makedirs(save_path, exist_ok=True)

    contact_net.train()

    for epoch in range(epoches_first):
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epoches_first}",
            unit="batch",
            ncols=120
        )

        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in train_pbar:
            contact, data_fcn_2, combined_pairwise, data_length, seq_raw, name, _ = batch

            contacts_batch = contact.float().to(device)  # [B, L, L]
            combined_pairwise = combined_pairwise.float().to(device)
            data_length = data_length.to(device)

            preds = contact_net(combined_pairwise)
            target_size = preds.shape[-1]

            gt_scaled, mask_scaled = build_scaled_mask_and_gt(
                contacts_batch, data_length, target_size, device
            )

            raw_loss_map = criterion_asl(preds, gt_scaled)
            valid_loss = (raw_loss_map * mask_scaled).sum()

            total_loss = valid_loss / mask_scaled.sum().clamp(min=1.0)
            main_loss_display = total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                contact_net.parameters(),
                max_norm=cfg.get("grad_clip", 1.0)
            )
            optimizer.step()

            scheduler.step()

            epoch_loss_sum += main_loss_display
            epoch_steps += 1

            current_step_lr = optimizer.param_groups[0]["lr"]
            train_pbar.set_postfix({
                "Loss": f"{main_loss_display:.5f}",
                "LR": f"{current_step_lr:.2e}"
            })

        avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)

        torch.save(
            contact_net.state_dict(),
            os.path.join(save_path, "fwufold_latest.pt")
        )

        if val_loaders is not None and len(val_loaders) > 0:
            print(f"\n[{'=' * 15} Epoch {epoch + 1} Validation {'=' * 15}]")
            print(f"Avg Train Loss: {avg_epoch_loss:.5f}")

            f1_list, p_list, r_list = [], [], []

            for val_name, v_loader in val_loaders.items():
                avg_f1, avg_p, avg_r, _ = model_eval_all_test(contact_net, v_loader, dataset_name=val_name)
                print(f"  -> [Dataset: {val_name: <15}] F1: {avg_f1:.4f} | P: {avg_p:.4f} | R: {avg_r:.4f}")

                f1_list.append(avg_f1)
                p_list.append(avg_p)
                r_list.append(avg_r)

            mean_f1 = float(np.mean(f1_list))
            mean_p = float(np.mean(p_list))
            mean_r = float(np.mean(r_list))

            print("-" * 55)
            print(f"  => [Overall Average  ] F1: {mean_f1:.4f} | P: {mean_p:.4f} | R: {mean_r:.4f}")
            print("=" * 55 + "\n")

            torch.save(
                contact_net.state_dict(),
                os.path.join(save_path, f"fwufold_ep{epoch + 1}_avgf1_{mean_f1:.3f}.pt")
            )

            contact_net.train()

    print("Training complete.")

def model_eval_all_test(contact_net, test_generator, dataset_name=""):
    device = next(contact_net.parameters()).device
    contact_net.eval()

    result_no_train = []
    result_nc = []

    desc_str = f"Eval {dataset_name}" if dataset_name else "Evaluating"
    test_pbar = tqdm(test_generator, desc=desc_str, leave=False, unit="batch")

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

    device = torch.device(
        f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────
    print("===== Loading Datasets =====")
    print(f"Save Dir:       {cfg['save_dir']}")
    print(f"Train Datasets: {cfg['train_files']} | Roots: {cfg['train_roots']}")
    print(f"Val Datasets:   {cfg['val_files']}   | Roots: {cfg['val_roots']}")

    dataset_train = RNADataset(
        cfg["train_roots"],
        dataset_name="_".join(cfg["train_files"]),
        cache_dir=cfg["cache_train"],
        is_train=True,
        is_test=False,
        upsample=True
    )

    sampler_train = BucketBatchSampler(dataset_train, batch_size=cfg["batch_size"], shuffle=True)
    loader_train = data.DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    val_loaders = {}
    for val_root, val_file in zip(cfg["val_roots"], cfg["val_files"]):
        dataset_val = RNADataset(
            [val_root],
            dataset_name=val_file,
            cache_dir=cfg["cache_val"],
            is_train=False,
            is_test=False,
            upsample=False
        )
        sampler_val = BucketBatchSampler(dataset_val, batch_size=1, shuffle=False)
        loader_val = data.DataLoader(
            dataset_val,
            batch_sampler=sampler_val,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        val_loaders[val_file] = loader_val

    contact_net = FWUNet(
        img_ch=cfg.get("img_ch", 17),
        output_ch=cfg.get("output_ch", 1),
        wave=cfg.get("wave", "haar")
    ).to(device)

    total_params = sum(p.numel() for p in contact_net.parameters())
    print(f"Model initialized | Params: {total_params:,} | "
          f"wave: {cfg.get('wave', 'haar')}")

    train(
        contact_net=contact_net,
        train_loader=loader_train,
        val_loaders=val_loaders,
        cfg=cfg
    )


if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple(
        'RNA_SS_data', 'data_fcn_2 seq_raw length name contact'
    )
    main()