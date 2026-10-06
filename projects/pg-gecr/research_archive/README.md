# Recovered study source bundle

These scripts and numerical outputs are copied from the original latent, updated ROI and v3.0 minor-revision supplements. They are retained as historical source, not silently relabelled as current primary results. The final expert-only primary metrics and 54/104 referral gate are in `minor_revision_results/expert_roi_referral_ablation.json`.

## Included
- Original cVAE training and four-hypothesis latent experiments (`latent_original/code` and `primary_code`).
- ROI feature extraction, model comparison, selected configurations and statistics scripts.
- Reviewer and minor-revision pHash, sensitivity, calibration, referral and learning-curve scripts/results.
- Auxiliary relative split manifest; training-only pseudo-label metadata with its original review warnings.
- Two historical NPZ prediction artifacts (not model weights).
- Source dependency records, retained for audit.

## Runtime boundary
Original scripts use historical `/mnt/data/...` locations and may execute training on import. Inspect and configure their paths before running them. Do not import them as a package. `historical_environment.txt` records the old environment; `python==...` is not a pip package requirement and must not be passed directly to pip.

Use the portable `train_roi.py` and `prepare_features.py` one directory above for expert-only training/export. They preserve the archived ROI extractor and locked model configuration, while adding explicit inputs and serialized model outputs. New runs are new experiments, not recovered original weights.

The historical model search includes exploratory test metrics; use locked configurations and validation-only choices for new work. Historical scripts and reports are not uniformly updated to the final evidence framing.

## Absent from recovered archives
No `.pt`, `.pth`, `.ckpt`, `.joblib` or `.pkl` trained weights were found. No `roi_results_fast/features.npz`, final expert-ROI per-image probabilities, or primary clinical image archive was recovered. Raw clinical images and manuscript/representative-image figures are not redistributed. Images require an authorized local dataset; auxiliary images do not reproduce primary outcomes.
