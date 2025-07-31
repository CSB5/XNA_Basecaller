<div align="center">
    <img src="./overview.png", width="600">
</div>

# XNA Basecaller

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--62347--z-brightgreen)](https://doi.org/10.1038/s41467-025-62347-z)
[![DOI](https://zenodo.org/badge/893805606.svg)](https://doi.org/10.5281/zenodo.15751758)

Code for paper: "Direct high-throughput deconvolution of non-canonical bases via nanopore sequencing and bootstrapped learning".

In our work we show how to achieve high-throughput sequencing of DNA containing Non-Canonical Bases (NCBs), a.k.a Unnatural Bases (UBs), using Nanopore and de novo basecalling enabled by spliced-based data-augmentation. The code here contains a basecaller architecture modified for learning to also basecall 1 or 2 additional UBs, and includes real-time data-augmentation for generating train data with UBs in all possible sequencing contexts.

More details in our Nature Communications publication: https://www.nature.com/articles/s41467-025-62347-z

## Setup

### Full Setup Script

Run the following commands to go through all setup steps at once with a single script (includes downloading 11GB of data). Refer to the detailed instructions further below to better understand the steps or for debugging.

```bash
git clone https://github.com/CSB5/XNA_Basecaller.git
cd XNA_Basecaller/
bash full_setup.sh
```

### Installation (stepwise)

Full installation should take only a few minutes (<5 mins).

Clone repository and enter directory:

```bash
git clone https://github.com/CSB5/XNA_Basecaller.git
cd XNA_Basecaller/
```

Download Minimap or create a symbolic link to it at `bin/minimap2`:

```bash
curl -L https://github.com/lh3/minimap2/releases/download/v2.17/minimap2-2.17_x64-linux.tar.bz2 | tar -jxvf - minimap2-2.17_x64-linux/minimap2
mv -v minimap2-2.17_x64-linux bin
```

Install python enviroment:

```bash
conda env create -f env.yml
conda activate xna_bc
```

Install UB-Bonito. Recommended to install with xna_bc activated to ensure valid python version (v3.9) is utilized.

```bash
cd ub-bonito/
python3 -m venv venv3
source venv3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# pip install -r requirements-cuda113.txt --extra-index-url https://download.pytorch.org/whl/cu113/
python setup.py develop
deactivate
cd ..
```

<ins>_NOTE_</ins>: Installation file "requirements.txt" is for Cuda v10, for Cuda v11 use "requirements-cuda113.txt".

### Download Data

Run `bash ./download_data.sh` script from the project root directory (`XNA_Basecaller/`) to download all the necessary files (Total 11GB). Might take several minutes (>10 mins):

- XNA basecaller baseline model (pre-trained with DNA only)
	- `XNA_Basecaller/ub-bonito/bonito/models/xna_r9.4.1_e8_sup@v3.3/`
- Evaluation reads from POC and Complex libraries (fast5 files)
	- `XNA_Basecaller/xna_libs/POC/reads/`
	- `XNA_Basecaller/xna_libs/CPLX/reads/`
- Pre-processed DNA and XNA train data files (e.g. chunks, references and segmentations)
	- DNA:
		- `XNA_Basecaller/ub-bonito/bonito/data/dna_r9.4.1/sampled_0.01/`
		- `XNA_Basecaller/ub-bonito/bonito/data/dna_r9.4.1/sampled_0.25/`
	- XNA:
		- `XNA_Basecaller/ub-bonito/bonito/data/xna_r9.4.1/`
		- `XNA_Basecaller/ub-bonito/bonito/data/xna_r9.4.1-sampled/`

<ins>_NOTE_</ins>: These files are kept to a minimum to reproduce our basecaller results, including pre-processed data used for training and a subset of the sequenced reads used for evaluation.
The complete set of our nanopore sequencing data is available from the European Nucleotide Archive (ENA) under project accession number [PRJEB82716](https://www.ebi.ac.uk/ena/browser/view/PRJEB82716).
Please refer to the end of this document for linux commands to download all the raw data from ENA server.

## Usage

In order to run the following scripts, make sure ub-bonito venv3 is **deactivated** (`deactivate`) and 
the correct conda enviroment is activated (`conda activate xna_bc`).

### Training

Use `train_and_eval.sh` to run basecaller training and evaluation. See file for more options available.

Quick runs (around 20 mins or less):

- Fully Synthetic - ~25% UB Acc.:
    - `bash ./train_and_eval.sh POC training/fully_synth-ubs_X-data_0.01-ub_prop_0.10 ub-bonito/bonito/data/dna_r9.4.1/sampled_0.01/ 0.10 -u X -b 98 -e 1 -Z`
- Hybrid - ~10% UB Acc.:
    - `bash ./train_and_eval.sh POC training/hybrid-ubs_X-data_0.01-ub_prop_0.10 ub-bonito/bonito/data/dna_r9.4.1/sampled_0.01/ 0.10 -u X -b 98 -e 1`
- Spliced - ~15% UB Acc.:
    - `bash ./train_and_eval.sh POC training/spliced-ubs_X-data_0.01-ub_prop_0.10 ub-bonito/bonito/data/dna_r9.4.1/sampled_0.01/ 0.10 -u X -b 98 -e 1 -m per_kmer`

Final models (should take a few hours), 70-80% UB Acc.:

```bash
for UBS in X Y XY; do
bash ./train_and_eval.sh CPLX training/spliced-ubs_$UBS-data_0.25-ub_prop_0.09-unfr_3 ub-bonito/bonito/data/dna_r9.4.1/sampled_0.25/ 0.09 -u $UBS -W -b 98 -m per_kmer -f -F 3 -E POC;
done
```

<ins>_NOTE_</ins>: Spliced method real-time train data generation requires many workers to avoid being the bottleneck depending on the batch size chosen. For faster training it is recommended to use more CPUs and larger batch size (arguments `-w 32 -b 512`).

### Evaluation

Use `eval_model.sh` to run ad-hoc basecaller evaluation. See file for more options available.
- `bash ./eval_model.sh POC training/fully_synth-ubs_X-data_0.01-ub_prop_0.10 -b 98 -u X`

### Additional Tools

Compare basecalling performances `src/tools/comp_basecalls_perf.py`
- Ex: `python src/tools/comp_basecalls_perf.py training/*`
- Use `-d` argument to output detailed performance per template and UB position

Segment train data chunks `src/tools/dtw_segmentation.py`
- Employs DTW to group signals from the same kmer based on the signal reference model
- Outputs "breakpoints.npy", file required by the train data generator embedded in ub-bonito
- Tool used to estimate breakpoints from pre-processed DNA and XNA train data
- Ex: `python src/tools/dtw_segmentation.py ub-bonito/bonito/data/dna_r9.4.1/sampled_0.01/ -p --pool_chunksize 1`

## Results

Final model performance on POC library:

Method | UB(s) | UB(s) Acc. | DNA Acc.
-- | -- | -- | --
Spliced | X | 77% | 91%
Spliced | Y | 81% | 92%
Spliced | XY | 71% | 92%

## Data availability (ENA)

The raw data from our five nanopore sequencing runs are available from the European Nucleotide Archive (ENA) under project accession number [PRJEB82716](https://www.ebi.ac.uk/ena/browser/view/PRJEB82716).
The archive files (.tar.gz) can be downloaded with the following commands:

```bash
# Approximately  90 GB
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR143/ERR14368252/POC-XNA_DNA-01_16.tar.gz
# Approximately 101 GB
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR143/ERR14369514/POC-DNA-17_20.tar.gz
# Approximately  72 GB
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR143/ERR14368253/POC-XNA-17_20.tar.gz
# Approximately  40 GB
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR143/ERR14369515/CPLX-XNA-all.tar.gz
# Approximately 157 GB
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR143/ERR14369516/CPLX-XNA-replenishment.tar.gz
```

## (Potential) Future updates

- UB kmer modeling script and tools
- End-to-end framework/pipeline description and/or script
- More utility tools

## Citation

Perez, M., Kimoto, M., Rajakumar, P. _et al_. Direct high-throughput deconvolution of non-canonical bases via nanopore sequencing and bootstrapped learning. _Nat Commun_ **16**, 6980 (2025). https://doi.org/10.1038/s41467-025-62347-z
