# Evidence and model card

## Intended use
Research and engineering review of an otoscopic AI study and its selective-referral policy. The portable policy consumes probability vectors. The bundle also provides ROI feature extraction/training/export on authorized local data; it does not ship the original trained image classifier. It is not a clinical decision service.

## Publication and sources
- Singh, Ankit Kumar; Raghuvanshi, Ajay Singh; Mehta, Rupa. *PG-GECR: A Provenance-Gated Hybrid Expert System for Auditable Otoscopic Classification and Human-Supervised Referral.* Array (2026).
- Publisher record: https://www.sciencedirect.com/science/article/pii/S2590005626006193
- Aggregate source: `PG_GECR_MinorRevision_Reproducibility_v3.0.zip`, `repro_v3_build/minor_revision_results/`.
- Metric framing checked against the final production manuscript. The primary classifier is **expert-only ROI**; older fusion results are not relabelled as primary.
- DOI, volume and article number are omitted until verified from publisher metadata.

## Primary population and evaluation
The source expert cohort contained 699 images. Conflict/duplicate exclusions left 687: 480 train, 103 validation and 104 test. Primary labels are AOM, ASOM, CSOM and Normal. Patient identifiers were unavailable; duplicate cleaning cannot prove patient independence. No patient images, names or per-image clinical records are redistributed here.

Primary expert-only ROI test metrics: accuracy 0.78846, balanced accuracy 0.68791, macro F1 0.67730, macro AUC 0.86200, MCC 0.59765. These are study-reported values, not independently recomputed from model inference in this companion.

## Referral evidence
The exploratory ROI-plus-latent gate retained 54/104 cases and correctly classified 51/54; 19/22 observed errors were referred. At the same post hoc coverage, ROI-only retained 48/54 correct and referred 16/22 errors. Uncertainty intervals and paired descriptive differences remain in the source JSON. Test reuse and image-level resampling limit interpretation.

**Minority-class limitation:** only seven CSOM test images were available. Primary CSOM recall was 2/7; neither of the two retained CSOM cases under the full gate was correct. Overall retained accuracy cannot be treated as safe clinical performance.

## Confounding and rejected evidence
ASOM images were structurally associated with JPEG acquisition, while AOM and CSOM were PNG. Decoding files does not remove compression or device signatures. Generated-image augmentation was rejected by validation; pseudo-label fusion did not establish a statistically supported incremental classification benefit. The cVAE is a complementary latent evidence view, not the primary accuracy driver.

## Separate auxiliary experiment
The 3,000-image auxiliary archive had 2,922 exact-unique images and its own five-class taxonomy. COM is not relabelled CSOM, and auxiliary data are not pooled into primary performance claims. Five repeated balanced subsets gave macro F1 0.76093 +/- 0.02102 at 25 training images/class and 0.93659 +/- 0.00715 at 350/class. These are sample-support analyses, not evidence of primary clinical generalization. The global cross-class conflict CSV is empty (header only); inspecting that output is not rerunning the pHash audit.

## What the local audit establishes
SHA-256 agreement with the copied source files; agreement between aggregate classwise counts and referral summaries; presence of the disclosed CSOM failures. It cannot establish scientific validity, recover missing model checkpoints, or independently validate the clinical outcomes.
