# FWUFold: A Frequency-Wavelet Enhanced U-Net Architecture for Accurate RNA Secondary Structure Prediction.
## Abstract
The precise prediction of RNA secondary structures is vital for unlocking their biological functions, yet conventional models face a major performance bottleneck when dealing with complex long-range dependencies and multi-scale features. To bridge this gap, we introduce FWUFold, a cutting-edge deep learning framework. FWUFold fuses the Discrete Wavelet Transform (DWT) into the pipeline of the classic U-Net. By breaking down intricate RNA sequences into multi-resolution frequency components, it vastly empowers the model to capture sharp structural boundaries and long-distance base pairs.

## Prerequisites
--python >= 3.9

--torch == 2.0.0 with cuDNN == 8.5.0 (CUDA == 11.7)

--pytorch_wavelets == 1.3.0

--numpy == 2.4.6

--pandas == 3.0.3

--scikit-learn == 1.8.0

--tqdm == 4.67.1

## Installation
### 1.Clone the repository.

```
git clone https://github.com/HpuBioinformatics/FWUFold.git
```
Navigate to the root of this repo and setup the conda environment.

### 2.Use the following command to create the environment.
```
conda env create -f environmnet.yaml
```
### 3.Activate conda environment.
```
conda activate FWUFold
``` 

## Usage

### Training
You can train our model using pre-defined data.
```bash
python train.py --train_files bpRNA
                --val_files bpRNA
                --save_dir ./
                --gpu_id 0
```
Note: save_dir can be user-defined as the destination path for saving the weights directory.
### Evaluating 
We provide the test script for user to evaluate the prediction result using the following command:
```bash
python test.py --test_files bpRNA
               --weight_path ./trian_alldata.pt
               --gpu_id 0 
```
### Predicting
We support two prediction modes depending on your input format: using a FASTA file (fasta_file) or a plain text sequence (raw_seq).
```bash
python predict.py --fasta_file ./data/my_sequences.fasta --weight_path ./trian_alldata.pt --output_dir ./my_results --gpu_id 0
```
```bash
python predict.py --raw_seq "GGGCCCGUAGUCUCAGGGUAAGAGCACACGCUGAAGUGUGUGGGUCGGCAGUUCGAUCCCGCUGCGGCCCACCA" --seq_name "tRNA_example" --weight_path ./trian_alldata.pt --output_dir ./my_results --gpu_id 0
```
Note: fasta_file is the directory for your predictions, which contains the FASTA files to be processed.
## Acknowledgements
This project draws inspiration from FWUFold. We extend our gratitude to the authors for their outstanding research and code, and we hope that readers will find their contributions equally valuable.

