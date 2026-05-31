import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import random
import os

label_dict = {
    '.': np.array([1, 0, 0]),
    '(': np.array([0, 1, 0]),
    ')': np.array([0, 0, 1])
}
seq_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),  # T or U
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),

    'R': np.array([1, 0, 0, 1]),
    'Y': np.array([0, 1, 1, 0]),
    'K': np.array([0, 1, 0, 1]),
    'M': np.array([1, 0, 1, 0]),
    'S': np.array([0, 0, 1, 1]),
    'W': np.array([1, 1, 0, 0]),
    'B': np.array([0, 1, 1, 1]),
    'D': np.array([1, 1, 0, 1]),
    'H': np.array([1, 1, 1, 0]),
    'V': np.array([1, 0, 1, 1]),
    'N': np.array([0, 0, 0, 0]),
    '_': np.array([0, 0, 0, 0]),
    '~': np.array([0, 0, 0, 0]),
    '.': np.array([0, 0, 0, 0]),
    'P': np.array([0, 0, 0, 0]),
    'I': np.array([0, 0, 0, 0]),
    'X': np.array([0, 0, 0, 0])
}

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('N')   # replace '.' to 'N'
        else:
            seq.append(char_dict[np.argmax(arr_row)])
    return ''.join(seq)


def contact_map_masks(data_lens, matrix_rep):
    n_seq = len(data_lens)
    assert matrix_rep.shape[0] == n_seq
    for i in range(n_seq):
        l = int(data_lens[i].cpu().numpy())
        matrix_rep[i, :l, :l] = 1
    return matrix_rep

# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:, 0].values)
    rnadata2 = list(data.loc[:, 4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1] > 0, rna_pairs))
    rna_pairs = (np.array(rna_pairs) - 1).tolist()
    return rna_pairs

# generate .dbn format
def generate_label_dot_bracket(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)


# extract the pseudoknot index given the data
def extract_pseudoknot(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                print(i, j)
                break


def find_pseudoknot(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                flag = True
                break
    return flag


def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x.upper()], str_list))
    # need to stack
    return np.stack(encoding, axis=0)


def struct_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: label_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)


def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_exact_new(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    # recall = tp/(tp+fn)
    # precision = tp/(tp+fp)
    # f1_score = 2*tp/(2*tp + fp + fn)
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    return precision, recall, f1_score

def build_scaled_mask_and_gt(
        contacts_full: torch.Tensor,
        data_length: torch.Tensor,
        target_size: int,
        device: torch.device,
):
    B, L_full, _ = contacts_full.shape

    # ── GT 下采样 ─────────────────────────────────────────────────────
    if target_size == L_full:
        gt_scaled = contacts_full
    else:
        gt_scaled = F.adaptive_max_pool2d(
            contacts_full.unsqueeze(1).float(),
            output_size=(target_size, target_size)
        ).squeeze(1)

    # ── 有效区域 mask（仅排除 padding） ───────────────────────────────
    scale_ratio = target_size / L_full
    mask_scaled = torch.zeros(B, target_size, target_size, device=device)
    for k, seq_len in enumerate(data_length):
        scaled_L = min(int(seq_len.item() * scale_ratio), target_size)
        mask_scaled[k, :scaled_L, :scaled_L] = 1.0

    return gt_scaled, mask_scaled

def read_fasta(fasta_file: str):
    """读取 FASTA 文件并返回 (名称, 序列) 列表"""
    samples = []
    with open(fasta_file, 'r') as f:
        name = ""
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    samples.append((name, "".join(seq)))
                name = line[1:] # 移除 '>'
                seq = []
            else:
                seq.append(line.upper())
        if name and seq:
            samples.append((name, "".join(seq)))
    return samples