# Raw data

Place authorized source files here. The preprocessing scripts expect:

```text
raw/
├── mimic-iii/
│   ├── PRESCRIPTIONS.csv
│   ├── DIAGNOSES_ICD.csv
│   ├── PROCEDURES_ICD.csv
│   └── ADMISSIONS.csv
├── mimic-iv/
│   ├── prescriptions_filtered.csv
│   ├── diagnoses_icd.csv
│   ├── procedures_icd.csv
│   └── admissions.csv
├── idx2SMILES.pkl
├── ndc2atc_level4.csv
├── drug-atc.csv
├── ndc2rxnorm_mapping.txt
└── drug-DDI.csv
```

The repository does not redistribute MIMIC data.
