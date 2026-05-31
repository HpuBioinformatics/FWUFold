import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import List
import collections
import _pickle as cPickle
from random import choices
from common.data_utils import read_fasta,label_dict,seq_encoding

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'data_fcn_2 seq_raw length name contact')

def make_dataset(directory: str) -> List[str]:
    instances = []
    directory = os.path.expanduser(directory)
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):
            if fname.endswith('.cPickle') or fname.endswith('.Pickle'):
                instances.append(os.path.join(root, fname))
    return instances

def pairs2map(pairs, seq_len):
    contact = np.zeros([seq_len, seq_len])
    for pair in pairs:
        i, j = pair
        if 0 <= i < seq_len and 0 <= j < seq_len:
            contact[i, j] = 1
    return contact

def Gaussian(x):
    return math.exp(-0.5 * (x * x))

def paired(x, y):
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0

def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient += score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient += score * Gaussian(add)
                    else:
                        break
            mat[i, j] = coefficient
    return mat

def get_relative_pos_encoding(L, d_model=16):
    coords = torch.arange(L).float()
    grid_i, grid_j = torch.meshgrid(coords, coords, indexing='ij')
    relative_pos = grid_i - grid_j

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    rel_pos_expanded = relative_pos.unsqueeze(-1) * div_term
    pe_sin = torch.sin(rel_pos_expanded)
    pe_cos = torch.cos(rel_pos_expanded)

    pe = torch.cat([pe_sin, pe_cos], dim=-1)
    pe = pe.permute(2, 0, 1).unsqueeze(0)
    return pe

def get_rbf_pos_encoding(L, d_model=16):
    coords = torch.arange(L).float()
    grid_i, grid_j = torch.meshgrid(coords, coords, indexing='ij')
    dist = torch.abs(grid_i - grid_j)
    mu = torch.linspace(0, L, d_model).view(d_model, 1, 1)
    sigma = L / (d_model * 0.5)
    pe = torch.exp(-0.5 * ((dist - mu) / sigma) ** 2)
    return pe.unsqueeze(0)


class FASTARNADataset(data.Dataset):

    def __init__(self, fasta_file: str = None, raw_seq: str = None, seq_name: str = "Test_seq", bucket_boundaries=[160, 320, 480, 640]):
        self.samples = []

        if fasta_file is not None and os.path.isfile(fasta_file):
            for name, seq in read_fasta(fasta_file):
                self.samples.append((name, seq.upper(), len(seq)))

        if raw_seq is not None:
            self.samples.append((seq_name, raw_seq.upper(), len(raw_seq)))

        if len(self.samples) == 0:
            raise ValueError("ValueError: Dataset initialization failed: active 'fasta_file' or 'raw_seq' is required!")

        self.bucket_info = self._build_buckets(bucket_boundaries)

    def _build_buckets(self, bucket_boundaries):
        buckets = [[] for _ in range(len(bucket_boundaries) + 1)]
        for idx, item in enumerate(self.samples):
            length = item[2]
            placed = False
            for i, boundary in enumerate(bucket_boundaries):
                if length <= boundary:
                    buckets[i].append(idx)
                    placed = True
                    break
            if not placed:
                buckets[-1].append(idx)
        return buckets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, seq_raw, length = self.samples[idx]

        data_fcn_2 = seq_encoding(seq_raw)

        pairwise_mat = creatmat(list(data_fcn_2))

        dummy_contact = np.zeros((length, length), dtype=np.int64)

        return (torch.tensor(dummy_contact).long(),
                torch.tensor(data_fcn_2).float(),
                torch.tensor(pairwise_mat).unsqueeze(0).float(),
                length,
                seq_raw,
                name)

class RNADataset(data.Dataset):
    def __init__(self, data_root: List[str], dataset_name: str, cache_dir="cache/",
                 upsample=False, upsample_pdb=False,
                 max_len=640,
                 is_train=True, is_test=False,
                 bucket_cache_file="bucket_info.npy"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.upsample = upsample
        self.upsample_pdb = upsample_pdb
        self.max_len = max_len
        self.is_train = is_train
        self.is_test = is_test

        self.samples = []
        index_cache_path = os.path.join(self.cache_dir, f"{dataset_name}_index.npy")

        if os.path.exists(index_cache_path):
            print(f"Loading index cache... {index_cache_path}")
            all_samples = np.load(index_cache_path, allow_pickle=True).tolist()
            self.samples = [s for s in all_samples if s[2] <= self.max_len]
            print(f"Remaining samples after filtering: {len(self.samples)} (max_len<={self.max_len})")
        else:
            print("Building index (scanning files)...")
            files = []
            for root in data_root:
                if isinstance(root, list):
                    for r in root:
                        files += make_dataset(r)
                else:
                    files += make_dataset(root)
            for f in files:
                with open(f, 'rb') as fo:
                    data_list = cPickle.load(fo, encoding='latin1')
                for i, sample in enumerate(data_list):
                    if sample.length <= self.max_len:
                        self.samples.append((f, i, sample.length, sample.name))
            np.save(index_cache_path, np.array(self.samples, dtype=object))
            print(f"Index built successfully. Number of valid samples: {len(self.samples)}")

        if self.upsample or self.upsample_pdb:
            self.samples = self.upsampling_data()

        prefix = f"{dataset_name}_{'train' if is_train else 'test'}_"
        self.bucket_cache_file = os.path.join(self.cache_dir, f"{prefix}{bucket_cache_file}")

        if os.path.exists(self.bucket_cache_file):
            self.bucket_info = np.load(self.bucket_cache_file, allow_pickle=True).tolist()
        else:
            self.bucket_info = self._build_buckets()
            np.save(self.bucket_cache_file, np.array(self.bucket_info, dtype=object))

    def _build_buckets(self, bucket_boundaries=[160, 320, 480, 640]):
        buckets = [[] for _ in range(len(bucket_boundaries) + 1)]
        for idx, item in enumerate(self.samples):
            length = item[2]
            placed = False
            for i, boundary in enumerate(bucket_boundaries):
                if length <= boundary:
                    buckets[i].append(idx)
                    placed = True
                    break
            if not placed:
                buckets[-1].append(idx)
        return buckets

    def upsampling_data(self):
        pdb_data_list = []
        normal_augment_list = []
        final_data_list = self.samples.copy()

        for item in self.samples:
            file_path = item[0]
            length = item[2]

            is_pdb = "PDB" in file_path

            if is_pdb:
                if self.upsample_pdb:
                    pdb_data_list.append(item)
            else:
                if self.upsample and 160 < length <= 640:
                    normal_augment_list.append(item)

        if pdb_data_list:
            print(f"PDB data augmentation enabled. Found {len(pdb_data_list)} items, upsampling by 4x...")
            pdb_augment = choices(pdb_data_list, k=4 * len(pdb_data_list))
            final_data_list.extend(pdb_augment)

        if normal_augment_list:
            print(f"Regular data augmentation enabled. Found {len(normal_augment_list)} items, upsampling by 3x...")
            normal_augment = choices(normal_augment_list, k=3 * len(normal_augment_list))
            final_data_list.extend(normal_augment)

        return final_data_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, sample_idx, length, name = self.samples[idx]

        cache_file = os.path.join(self.cache_dir, f"{name}.npz")

        cache_dir_path = os.path.dirname(cache_file)
        if not os.path.exists(cache_dir_path):
            try:
                os.makedirs(cache_dir_path, exist_ok=True)
            except OSError:
                pass

        if os.path.isfile(cache_file):
            try:
                cached = np.load(cache_file, allow_pickle=True)
                contact = cached['contact']
                data_fcn_2 = cached['data_fcn_2']
                pairwise_mat = cached['pairwise_mat']
                seq_raw = str(cached['seq_raw'])
            except:
                return self._generate_and_cache(file_path, sample_idx, cache_file, length)
        else:
            return self._generate_and_cache(file_path, sample_idx, cache_file, length)

        return (torch.tensor(contact).long(),
                torch.tensor(data_fcn_2).float(),
                torch.tensor(pairwise_mat).unsqueeze(0).float(),
                length,
                seq_raw,
                name)

    def _generate_and_cache(self, file_path, sample_idx, cache_file, length):
        with open(file_path, 'rb') as fo:
            full_list = cPickle.load(fo, encoding='latin1')
            sample = full_list[sample_idx]

        contact = pairs2map(sample.contact, sample.length)
        data_fcn_2 = sample.data_fcn_2
        pairwise_mat = creatmat(data_fcn_2)
        seq_raw = sample.seq_raw

        np.savez(cache_file,
                 contact=contact,
                 data_fcn_2=data_fcn_2,
                 pairwise_mat=pairwise_mat,
                 seq_raw=seq_raw,
                 length=length)

        return (torch.tensor(contact).long(),
                torch.tensor(data_fcn_2).float(),
                torch.tensor(pairwise_mat).unsqueeze(0).float(),
                length,
                seq_raw,
                sample.name)

class BucketBatchSampler(data.Sampler):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buckets = dataset.bucket_info

    def __iter__(self):
        all_batches = []
        for bucket in self.buckets:
            if not bucket: continue
            bucket_lengths = [self.dataset.samples[i][2] for i in bucket]
            sorted_bucket = [x for _, x in sorted(zip(bucket_lengths, bucket), key=lambda pair: pair[0])]
            for i in range(0, len(sorted_bucket), self.batch_size):
                batch = sorted_bucket[i:i + self.batch_size]
                all_batches.append(batch)
        if self.shuffle: np.random.shuffle(all_batches)
        for batch in all_batches: yield batch

    def __len__(self):
        return sum(math.ceil(len(bucket) / self.batch_size) for bucket in self.buckets)

def pad_tensor(tensor_list, max_len, dim=0):
    padded = []
    for t in tensor_list:
        shape = t.shape
        pad_args = []
        if len(shape) == 2 and shape[1] == 4:
            pad_args = (0, 0, 0, max_len - shape[0])
        elif len(shape) == 3:
            pad_args = (0, max_len - shape[2], 0, max_len - shape[1])
        elif len(shape) == 2 and shape[0] == shape[1]:
            pad_args = (0, max_len - shape[1], 0, max_len - shape[0])

        if pad_args:
            t = F.pad(t, pad_args, value=0)
        padded.append(t)
    return torch.stack(padded, dim=dim)

def collate_fn(batch):
    contact_list, data_fcn_2_list, pairwise_list, length_list, seq_raw_list, name_list = zip(*batch)

    batch_max_len = max(length_list)
    set_max_len = math.ceil(batch_max_len / 16) * 16

    contact = pad_tensor(contact_list, set_max_len, dim=0)
    data_fcn_2 = pad_tensor(data_fcn_2_list, set_max_len, dim=0)
    pairwise_mat = pad_tensor(pairwise_list, set_max_len, dim=0)

    seq_trans = data_fcn_2.transpose(1, 2)
    kron_tensor = torch.einsum('bci,bdj->bcdij', seq_trans, seq_trans)
    kron_tensor = kron_tensor.reshape(contact.shape[0], 16, set_max_len, set_max_len)

    if pairwise_mat.dim() == 3:
        pairwise_mat = pairwise_mat.unsqueeze(1)

    combined_pairwise = torch.cat([kron_tensor, pairwise_mat], dim=1)

    data_length = torch.tensor(length_list, dtype=torch.long)
    seq_raw = list(seq_raw_list)
    name = list(name_list)

    return contact, data_fcn_2, combined_pairwise, data_length, seq_raw, name, set_max_len

