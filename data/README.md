# data/

This folder contains EEG datasets used in Phase 2 evaluation.

## Structure

```
data/
├── eegdenoisenet/          # EEGdenoiseNet benchmark dataset
│   ├── EEG_all_epochs.mat  # 4514 clean 1-s EEG epochs (from Matlab)
│   └── README.txt
├── chbmit_sample/          # CHB-MIT Scalp EEG sample (3 subjects)
│   ├── chb01_01.edf        # Raw EDF recording
│   └── chb01_01.edf.seizures
└── dataset.zip             # Original compressed archive (source data)
```

## How to set up

1. Unzip `dataset.zip` into this folder:
   ```bash
   cd data
   unzip ../dataset.zip -d .
   ```

2. For EEGdenoiseNet: download from https://github.com/ncclabsustech/EEGdenoiseNet
   - Place `EEG_all_epochs.mat` in `data/eegdenoisenet/`

3. For CHB-MIT: available at https://physionet.org/content/chbmit/1.0.0/
   - Place selected `.edf` files in `data/chbmit_sample/`

## Notes

- All datasets are used for **validation only** — the model trains on synthetic data.
- Scripts auto-fall back to synthesised data if files are absent (see `evaluation/external_validation.py`).
- Raw EDF and MAT files are excluded from version control (`.gitignore`) due to size.
