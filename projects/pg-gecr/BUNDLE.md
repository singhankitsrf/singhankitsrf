# Training source and artifact bundle

## Included now

The [research archive](research_archive/) includes 74 recovered source/evidence files: original cVAE training, four-hypothesis encoding, synthetic-admission experiments, ROI feature extraction/classification, calibration and statistics, reviewer sensitivity analyses and split metadata. [File hashes](research_archive_manifest.json) preserve the recovered contents.

A portable expert-only training/export path is also supplied:

```bash
python -m pip install -r projects/pg-gecr/requirements-training.txt
python projects/pg-gecr/prepare_features.py --manifest /path/to/splits.csv --image-root /path/to/authorized-images --output /path/to/features.npz
python projects/pg-gecr/train_roi.py --features /path/to/features.npz --output /path/to/new-model
```

The manifest schema is `path,label,split,source`. Use image-root-relative paths; labels `AOM, ASOM, CSOM, Normal`; splits `train, validation, test`; source `expert`. Example schema only:

```csv
path,label,split,source
AOM/example.png,AOM,train,expert
```

All partitions and all four training classes must be provided. The extractor rejects exact duplicates and pseudo-labelled primary inputs; this does not replace patient grouping, expert adjudication or near-duplicate audits. To reproduce the original cohort, use the original authorized dataset and validated split order, not a fresh arbitrary manifest.

The trainer saves `expert_roi.joblib`, calibrated `probabilities.npz`, `model.json` and artifact checksums. It fits only on training data and fits temperature only on validation data. Those exported weights belong to a **new run** and are not the original published checkpoints.

## What still needs to be supplied

The recovered archives contain no trained `.pt`, `.pth`, `.ckpt`, `.joblib` or `.pkl` files. Historical prediction NPZ files are included but are not weights. For complete original-model reproduction/serving, supply:

1. Original cVAE checkpoint (`cvae.pt` or `cvae_state.pt`) and the matching architecture variant.
2. Serialized expert-only ROI classifier or the original ROI feature cache with split mapping.
3. Latent ensemble classifiers, calibration temperatures, preprocessing/version records and feature ordering.
4. Original expert-ROI probability artifact and manifest to replay the final referral ablation.
5. An authorized private path to primary images when retraining is needed.

The auxiliary five-class archive cannot replace the primary four-class cohort. The original latent scripts preserve historical paths and exploratory branches; their execution requires configuring the documented dependencies and inputs. The portable path covers expert-only ROI training/export, not a newly asserted full reproduction of cVAE/ensemble experiments.
