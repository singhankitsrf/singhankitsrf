AIIMS RAIPUR OTOSCOPIC DATASET — EXPANDED PSEUDO-LABELLED VERSION 1
=========================================================================

PURPOSE
-------
This package organizes the original 699 expert-labelled images and generates provisional AI pseudo-labels for the 204 images in the Testing dataset.

IMPORTANT CLINICAL LIMITATION
-----------------------------
The newly assigned labels are NOT expert ground truth. They are model-assisted pseudo-labels produced for research organization and ENT review. They must not be used for patient diagnosis, clinical decisions, or final Q1 manuscript claims until verified by qualified ENT specialists.

WHAT WAS CREATED
----------------
1. 01_Original_Expert_Labeled_Unmodified_699
   Exact organization of the 699 originally labelled images.

2. 02_Merged_Conservative_Expanded_736
   The original 699 images plus 37 unique, conservative AI pseudo-labelled PNG test images:
   - AOM: 3 added
   - CSOM: 2 added
   - Normal: 32 added
   - ASOM: 0 added because JPG acquisition format is confounded with the ASOM training class.
   Total: 736 images.

3. 03_QA_Clean_Research_Set_724
   A model-development version that removes:
   - 6 images participating in three AOM-versus-Normal cross-label duplicate conflicts;
   - 6 redundant copies from six same-label duplicate pairs;
   and adds the 37 conservative unique pseudo-labels.
   Total: 724 images.

4. 04_All_Test_AI_Predictions_For_ENT_Review
   All readable test images organized by predicted class and confidence tier. These remain pending clinical review.

5. 05_Excluded_or_Duplicate
   Contains the damaged JPG and the redundant pseudo-labelled duplicate.

6. 06_Manifests_and_QA
   CSV and JSON records containing every prediction, probability, model agreement, nearest labelled image, duplicate flags, dataset provenance, and cross-validation metrics.

7. 07_ENT_Review_Contact_Sheets
   Contact sheets arranged for rapid clinical review.

PSEUDO-LABELLING METHOD
-----------------------
Images were cropped to the visible otoscopic field and standardized to reduce black-border and resolution cues. A four-model ensemble was trained from colour, texture, local binary pattern, HOG, edge and sharpness features:
- PCA logistic regression
- PCA RBF support-vector machine
- Extra Trees
- LightGBM

Duplicate-aware grouped five-fold cross-validation was used. The weighted ensemble obtained:
- Accuracy: 0.833
- Balanced accuracy: 0.655
- Macro F1: 0.687
- Matthews correlation coefficient: 0.656

These values are internal pseudo-labelling quality indicators, not clinical validation.

CONSERVATIVE ACCEPTANCE RULE
----------------------------
A test image was added to the conservative expanded dataset only when:
- it was a PNG image;
- all four classifiers agreed;
- ensemble confidence and top-two margin exceeded class-specific strict thresholds;
- it was readable;
- and it was not a duplicate of another accepted test image or an existing training image.

JPG test images were deliberately withheld from the merged dataset because all original ASOM images were JPG images, creating acquisition-source shortcut risk. They are retained in the ENT review folders with provisional predictions.

REQUIRED NEXT STEP
------------------
Two ENT specialists should independently review the manifest and contact sheets. Final labels should be recorded as:
- ENT reviewer 1 label
- ENT reviewer 2 label
- consensus/adjudicated label
- include/exclude decision

Only the adjudicated labels should be used as ground truth in the third Q1 manuscript.
