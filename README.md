# TDH-DR

Official implementation of **Taxonomy-Aware Dual-Mechanism Hypergraph Framework for Drug Recommendation (TDH-DR)**.

The repository contains the complete TDH-DR model, preprocessing pipelines for MIMIC-III and MIMIC-IV, the controlled ParentCat baseline, the no-drug-representation ablation, and the medication RHGNN-to-HGNN compatibility experiment.

## 1. Data preparation

The preprocessing pipeline follows the data organization and mapping resources used by **CARMEN**:

- CARMEN repository: https://github.com/bit1029public/Carmen
- CARMEN data directory: https://github.com/bit1029public/Carmen/tree/main/data

MIMIC data are **not distributed with this repository**. Users must obtain authorized access from PhysioNet and download the required clinical tables themselves.

### 1.1 Raw MIMIC files

Place the raw clinical tables under `data/raw/` using the following layout:

```text
data/raw/
├── mimic-iii/
│   ├── PRESCRIPTIONS.csv
│   ├── DIAGNOSES_ICD.csv
│   ├── PROCEDURES_ICD.csv
│   └── ADMISSIONS.csv
└── mimic-iv/
    ├── prescriptions_filtered.csv
    ├── diagnoses_icd.csv
    ├── procedures_icd.csv
    └── admissions.csv
```

For MIMIC-III, download the source tables from the authorized MIMIC-III distribution on PhysioNet.

For MIMIC-IV, download the corresponding source tables from the authorized MIMIC-IV distribution on PhysioNet. The current preprocessing script expects `prescriptions_filtered.csv`, consistent with the CARMEN MIMIC-IV preprocessing pipeline.

### 1.2 Auxiliary mapping and DDI files

The preprocessing scripts additionally require the following files:

```text
data/raw/
├── idx2SMILES.pkl
├── ndc2atc_level4.csv
├── drug-atc.csv
├── ndc2rxnorm_mapping.txt
└── drug-DDI.csv
```

These resources follow the CARMEN preprocessing setup. The CARMEN `data/` directory provides the corresponding resources, including:

```text
idx2SMILES.pkl
ndc2atc_level4.zip
ndc2rxnorm_mapping.zip
drug-atc.zip
drug-DDI.rar
```

Download them from the CARMEN repository, extract the archives where necessary, and place the resulting files in `data/raw/` with the names shown above.

### 1.3 Run preprocessing

From the repository root, run:

```bash
python src/processing_mimic_iii.py
python src/processing_mimic_iv.py
```

Each script performs the complete preprocessing workflow in one run, including clinical-table filtering, medication-code mapping, vocabulary construction, patient-record construction, DDI-matrix generation, and taxonomy-aware hypergraph construction. Dataset splitting is not performed during preprocessing. The experiment scripts load the complete patient-record file and perform the train/validation/test split at runtime using the original experimental split logic.

The generated files are written to `data/processed/`.

For MIMIC-III:

```text
data/processed/
├── voc_final.pkl
├── records_final.pkl
├── ddi_A_final.pkl
└── HG_matrix.pkl
```

For MIMIC-IV:

```text
data/processed/
├── voc_final_4.pkl
├── records_final_4.pkl
├── ddi_A_final_4.pkl
└── HG_matrix_4.pkl
```

### 1.4 Dataset split during experiments

The preprocessing scripts keep the complete patient-level record file intact. Each training/testing script performs the patient-level split after loading `records_final.pkl` (MIMIC-III) or `records_final_4.pkl` (MIMIC-IV), using the original experiment logic:

```text
training:   first 2/3 of patients
test:       first half of the remaining 1/3
validation: second half of the remaining 1/3
```

This corresponds to a 2/3 : 1/6 : 1/6 train/test/validation partition and ensures that all visits from the same patient remain in the same subset.

## 2. Environment

The code was tested with:

```text
Python 3.11
PyTorch 2.4.0
CUDA 12.1
```

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

CUDA toolkit and NVIDIA driver installation should be configured separately according to the local GPU environment.

## 3. Main experiments

Run all commands from the repository root.

### 3.1 MIMIC-III

```bash
python src/run_mimic_iii.py \
  --cuda 0 \
  --lr 5.9e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 512 \
  --alpha 0.94 \
  --beta 0.76 \
  --ddi \
  --use_hgd --use_hgp --use_hgm \
  --model_name TDHDR_mimic_iii_full
```

Test the selected checkpoint with:

```bash
python src/run_mimic_iii.py \
  --cuda 0 \
  --lr 5.9e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 512 \
  --alpha 0.94 \
  --beta 0.76 \
  --ddi \
  --use_hgd --use_hgp --use_hgm \
  --model_name TDHDR_mimic_iii_full \
  --Test
```

### 3.2 MIMIC-IV

```bash
python src/run_mimic_iv.py \
  --cuda 0 \
  --lr 1e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 256 \
  --alpha 0.95 \
  --beta 0.88 \
  --ddi \
  --use_hgd --use_hgp --use_hgm \
  --model_name TDHDR_mimic_iv_full
```

Test the selected checkpoint with:

```bash
python src/run_mimic_iv.py \
  --cuda 0 \
  --lr 1e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 256 \
  --alpha 0.95 \
  --beta 0.88 \
  --ddi \
  --use_hgd --use_hgp --use_hgm \
  --model_name TDHDR_mimic_iv_full \
  --Test
```

## 4. Ablation and controlled experiments

### 4.1 Hypergraph-module ablations

The full implementation in `src/model.py` supports diagnosis, procedure, and medication hypergraphs through:

```text
--use_hgd
--use_hgp
--use_hgm
```

Accordingly, `w/o G_d`, `w/o G_p`, `w/o G_m`, pairwise removals, and removal of all three hypergraphs can be reproduced by omitting the corresponding flags.

The DDI-loss ablation is reproduced by omitting `--ddi`.

### 4.2 Medication RHGNN -> regular HGNN

The compatibility experiment that replaces the medication RHGNN with a conventional HGNN uses:

```bash
python src/run_mimic_iii_ab_Dm_HGNN.py \
  --cuda 0 \
  --lr 5.9e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 512 \
  --alpha 0.94 \
  --beta 0.76 \
  --ddi \
  --use_hgd --use_hgp --use_hgm \
  --model_name TDHDR_mimic_iii_med_regular_hgnn
```

This script fixes `hgmtype=3`, while the remaining architecture is shared with `model.py`.

### 4.3 ParentCat baseline

ParentCat directly combines entity embeddings with learnable parent-category embeddings and therefore uses a separate model implementation.

MIMIC-III:

```bash
python src/run_mimic_iii_parent_category.py \
  --cuda 0 \
  --lr 5.9e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 512 \
  --alpha 0.94 \
  --beta 0.76 \
  --ddi \
  --parent_weight 0.5 \
  --model_name ParentCat_mimic_iii
```

MIMIC-IV:

```bash
python src/run_mimic_iv_parent_category.py \
  --cuda 0 \
  --lr 1e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 256 \
  --alpha 0.95 \
  --beta 0.88 \
  --ddi \
  --parent_weight 0.5 \
  --model_name ParentCat_mimic_iv
```

### 4.4 w/o explicit drug representation

This ablation removes the explicit medication-representation pathway and uses a separate implementation:

```bash
python src/run_mimic_iii_no_drugrep.py \
  --cuda 0 \
  --lr 5.9e-4 \
  --epoch 120 \
  --bs 8 \
  --dim 512 \
  --alpha 0.94 \
  --beta 0.76 \
  --ddi \
  --use_hgd --use_hgp \
  --model_name TDHDR_mimic_iii_no_drugrep
```

Do not enable the medication-hypergraph flags for this experiment.

## 5. Checkpoints

Training outputs are stored under:

```text
saved/<model_name>/
```

The selected checkpoint is stored as:

```text
saved/<model_name>/best.model
```

and is used by the corresponding `--Test` mode.

## 6. Project structure

```text
TDH-DR/
├── data/
│   ├── raw/                         # authorized MIMIC tables and auxiliary mapping files
│   └── processed/                   # generated preprocessing outputs
├── saved/                           # model checkpoints and training histories
├── src/
│   ├── processing_mimic_iii.py      # MIMIC-III preprocessing
│   ├── processing_mimic_iv.py       # MIMIC-IV preprocessing
│   ├── model.py                     # full TDH-DR and mergeable ablations
│   ├── model_parent_category.py     # ParentCat baseline
│   ├── model_no_drugrep.py          # no explicit drug representation
│   ├── hypergraph.py
│   ├── dataset.py
│   ├── util.py
│   ├── paths.py
│   ├── run_mimic_iii.py
│   ├── run_mimic_iv.py
│   ├── run_mimic_iii_ab_Dm_HGNN.py
│   ├── run_mimic_iii_parent_category.py
│   ├── run_mimic_iv_parent_category.py
│   └── run_mimic_iii_no_drugrep.py
├── requirements.txt
└── README.md
```

Plotting scripts are intentionally excluded from this repository.

## 7. Data-use note

MIMIC-III and MIMIC-IV are credentialed datasets distributed through PhysioNet. This repository does not redistribute raw or patient-level processed MIMIC data. Users are responsible for obtaining the required access and complying with the applicable PhysioNet data-use agreement.

## Acknowledgement

The preprocessing pipeline and auxiliary mapping resources follow the setup used by CARMEN:

**Context-aware Safe Medication Recommendations with Molecular Graph and DDI Graph Embedding**  
https://github.com/bit1029public/Carmen
