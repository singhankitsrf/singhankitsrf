# QA-724 GenAI Otoscopy Reproducibility Package

This package accompanies the 25-page manuscript **“A Generative AI–Augmented Deep Learning Expert System for Data-Efficient Detection and Referral Triage of Mucosal Middle-Ear Disease.”**

## What is included
- `manuscript/`: editable Word manuscript and rendered PDF.
- `code/`: executed Python pipelines for data QA, conditional VAE training, label-independent four-hypothesis latent encoding, classifier comparison, synthetic-data quality gating, calibration, referral analysis, and figure generation.
- `results/`: machine-readable result tables, calibrated predictions, confusion matrix, class report, bootstrap distributions, and optimization outputs.
- `figures/`: all programmatically generated manuscript figures.
- `metadata/`: pseudo-label manifest, grouped cross-validation metrics, and source QA notes.

## Dataset is not redistributed
Clinical image files are intentionally excluded. Place the authorized dataset at:

```
<project-root>/Dataset/Dataset/Training dataset/Otitis Media/
<project-root>/Dataset/Dataset/Testing dataset/
```

Expected expert-labelled folders are `AOM`, `ASOM`, `CSOM`, and `Normal Tympanic Membrane`.

## Environment
The recorded experiment ran on Linux with Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu, and no GPU. The seed was fixed at 42. Install dependencies with:

```
python -m pip install -r requirements.txt
```

## Running
Set the project root when needed:

```
export ESWA_QA724_ROOT=/path/to/project-root
python code/run_genai_experiment.py
python code/run_latent_experiment.py
python code/optimize_gate_fast.py
python code/make_figures.py
```

## Interpretation constraint
The 37 additions to QA-724 are conservative model-assisted pseudo-labels pending ENT adjudication. They are not independent clinical ground truth. The fixed expert-only validation and test sets exclude these images. The unrestricted cVAE augmentation did not pass the validation gate and was rejected; this negative result is intentionally retained.
