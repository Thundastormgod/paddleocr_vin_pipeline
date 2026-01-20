PaddleOCR VIN Recognition Pipeline
===================================

A complete OCR pipeline for Vehicle Identification Number (VIN) recognition 
from engraved metal plates, using PaddleOCR with specialized preprocessing 
and postprocessing.


KEY METRICS (Current Performance)
=================================

    Character-Level F1 Score:  55%  (baseline: 43%, improvement: +29%)
    Exact Match Rate:          25%  (96/382 images)
    Precision:                 57%
    Recall:                    54%

    Note: Industry target is 95%+ exact match, 98%+ F1.
    This pipeline establishes a baseline for further development.


EXPERIMENT SUMMARY
==================

Dataset
-------
- Total Images: 382 VIN plate images
- Source: DagsHub bucket (JRL-VIN project)
- Image Type: Engraved metal VIN plates from vehicles
- Ground Truth: Manual annotations with verified VINs


Industry Metrics Achieved
-------------------------

Metric                  Baseline        With Pipeline   Improvement   Industry Target
----------------------  --------------  --------------  ------------  ---------------
Exact Match Rate        5% (19/382)     25% (96/382)    +400%         95%+
Character-Level F1      43%             55%             +29%          98%+
Precision               45%             57%             +27%          98%+
Recall                  42%             54%             +29%          98%+
Detection Rate          99.7%           99.7%           --            99%+
Avg Processing Time     ~1.6s/image     ~2.3s/image     +44%          <5s


Additional Metrics to Explore
-----------------------------

Metric                      Formula                                 Status
--------------------------  --------------------------------------  --------------
CER (Character Error Rate)  (S + D + I) / N                         To calculate
NED (Normalized Edit Dist)  edit_distance / max(len_pred, len_gt)   To calculate
Word Error Rate (WER)       Errors at VIN level                     Have (1-exact match)
Per-Position Accuracy       Accuracy at each of 17 positions        To calculate
Levenshtein Distance (Avg)  Mean edits needed to correct            To calculate

Formulas:
- CER: (Substitutions + Deletions + Insertions) / Total Characters
- NED: Levenshtein(predicted, ground_truth) / max(len(predicted), len(ground_truth))
- Per-Position: correct_at_position_i / total_samples for i in 1-17


Why These Results Matter
------------------------

1. BASELINE PERFORMANCE GAP
   Raw PaddleOCR achieves only 5% exact match on engraved plates due to:
   - Metal surface reflections and lighting variations
   - Character confusions (O/0, I/1, S/5) common on stamped text
   - Artifact characters from plate borders and stamps

2. PIPELINE IMPROVEMENTS
   Our preprocessing + postprocessing pipeline achieves 5x improvement:
   - CLAHE contrast enhancement handles lighting variations
   - Artifact removal strips border characters (*, #, X prefixes)
   - Invalid character correction (I->1, O->0, Q->0 per VIN standard)
   - Position-based correction (digits in sequential section)

3. GAP TO PRODUCTION
   Current 25% exact match is NOT production-ready (industry requires 95%+).
   This baseline establishes:
   - A validated preprocessing approach that works for engraved plates
   - Identified failure modes for targeted improvements
   - A foundation for the team to build upon


Recommended Next Steps (NOT YET IMPLEMENTED)
--------------------------------------------

Priority   Action                                      Expected Impact     Status
---------  ------------------------------------------  ------------------  -----------
High       Fine-tune detection model on VIN plates    +20-30% exact match Not started
High       Train custom recognition model on charset  +15-25% exact match Not started
Medium     Implement confidence-weighted voting       +5-10% exact match  Not started
Medium     Add manufacturer-specific WMI validation   +3-5% exact match   Not started
Low        Multi-angle image capture                  +5-10% exact match  Not started


CLI TESTING RESULTS (January 2026)
==================================

Test Environment
----------------
- PaddleOCR Version: 3.x (PP-OCRv5)
- Preprocessing Mode: engraved (CLAHE + bilateral filter)
- Python: 3.12
- Platform: macOS (Apple Silicon)

Images Tested
-------------
1. 1-VIN_-_SAL119E90SA606112_.jpg
2. 10-VIN_-_SAL1A2A40SA606645_.jpg
3. 1000-VIN_-_SAL1P9EU2SA606633_.jpg
4. 1001-VIN_-_SAL1P9EU2SA606664_.jpg

Preprocessing Pipeline
----------------------
Step 1: Load image (BGR format)
Step 2: Convert to grayscale
Step 3: Apply CLAHE (clip_limit=2.0, tile_size=8x8)
Step 4: Bilateral filter (d=5, sigmaColor=50, sigmaSpace=50)
Step 5: Convert back to BGR (3-channel) for PaddleOCR

Model Configuration
-------------------
- Detection Model: PP-OCRv5_server_det
- Recognition Model: en_PP-OCRv5_mobile_rec
- Language: English
- text_det_box_thresh: 0.3

Test Results
------------

Image                               Expected VIN          Predicted VIN            Conf    Match
----------------------------------  --------------------  -----------------------  ------  -----
1-VIN_-_SAL119E90SA606112_.jpg      SAL119E90SA606112     2ESAL119E90SA606112      55%     No
10-VIN_-_SAL1A2A40SA606645_.jpg     SAL1A2A40SA606645     SAL1A2K40SR606E45M       69%     No
1000-VIN_-_SAL1P9EU2SA606633_.jpg   SAL1P9EU2SA606633     1401SA10EH/SA5066331     33%     No
1001-VIN_-_SAL1P9EU2SA606664_.jpg   SAL1P9EU2SA606664     SAL1P9EU2SA606664        96%     YES

Summary: 1/4 exact matches (25%) - consistent with full dataset results

Errors Encountered During Development
-------------------------------------

1. DEPRECATED PARAMETER ERROR
   Error: DeprecationWarning: det_db_box_thresh has been deprecated
   Fix: Changed to text_det_box_thresh in PaddleOCR 3.x

2. INVALID PARAMETER ERROR
   Error: ValueError: Unknown argument: rec_thresh
   Fix: Removed rec_thresh parameter (no longer supported in PaddleOCR 3.x)

3. IMAGE DIMENSION ERROR
   Error: ValueError: not enough values to unpack (expected 3, got 2)
   Cause: PaddleOCR expects 3-channel BGR images, preprocessing returned grayscale
   Fix: Added cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) after preprocessing

4. OCR ARTIFACT CHARACTERS
   Issue: Raw OCR output contains *, #, X, / characters from plate borders
   Example: "*SAL1P9EU2SA606664" -> "SAL1P9EU2SA606664"
   Note: "/" character not yet filtered (seen in test image 1000)

Observed Failure Modes
----------------------
- Prefix artifacts: "2E*", "I" prepended to VIN
- Character confusion: A<->K, 6<->E, 9<->0
- Slash insertion: "/" appearing mid-VIN from scratches/reflections
- Low confidence (<50%) correlates with incorrect predictions


INSTALLATION
============

    pip install -r requirements.txt


QUICK START
===========

    from vin_pipeline import VINOCRPipeline

    pipeline = VINOCRPipeline()
    result = pipeline.recognize('path/to/vin_image.jpg')

    print(result['vin'])           # "SAL1P9EU2SA606664"
    print(result['confidence'])    # 0.91
    print(result['raw_ocr'])       # "XSAL1P9EU2SA606664*"


PIPELINE ARCHITECTURE
=====================

    +---------------------------------------------------------------------+
    |                     VIN OCR PIPELINE                                |
    +---------------------------------------------------------------------+
    |  +------------+    +------------+    +----------------------+       |
    |  | PREPROCESS |--->| PADDLEOCR  |--->|   POSTPROCESSOR      |       |
    |  +------------+    +------------+    +----------------------+       |
    |  | - Grayscale|    | - PP-OCRv4 |    | - Artifact Removal   |       |
    |  | - CLAHE    |    | - Detection|    | - Invalid Char Fix   |       |
    |  | - Bilateral|    | - Recogn.  |    | - Checksum Validate  |       |
    |  +------------+    +------------+    +----------------------+       |
    +---------------------------------------------------------------------+


FILES
=====

    paddleocr_vin_pipeline/
    |-- README.md                 # This file
    |-- requirements.txt          # Dependencies
    |-- vin_pipeline.py           # Complete pipeline (single-file)
    |-- example_usage.py          # Usage examples
    |-- ARCHITECTURE.md           # Detailed architecture docs
    |-- LICENSE                   # Apache 2.0 License
    |-- NOTICE                    # Attribution notices
    |-- tests/
    |   +-- test_vin_pipeline.py  # Test suite (52+ tests)
    +-- results/
        |-- experiment_summary.json
        |-- detailed_metrics.json
        +-- sample_results.csv


CONFIGURATION
=============

    pipeline = VINOCRPipeline(
        preprocess_mode='engraved',  # 'none', 'fast', 'balanced', 'engraved'
        enable_postprocess=True,     # Enable VIN correction
        verbose=False                # Print processing steps
    )


CHARACTER CONFUSION HANDLING
============================

Confusion   Solution         Reason
----------  ---------------  ----------------------------------
I -> 1      Auto-fix         I is invalid in VIN
O -> 0      Auto-fix         O is invalid in VIN
Q -> 0      Auto-fix         Q is invalid in VIN
S <-> 5     Position-based   Prefer digits in sequential section
L <-> 1     Position-based   Check surrounding characters
* # X       Remove           Artifact characters


VIN FORMAT REFERENCE
====================

    Position:  1  2  3  | 4  5  6  7  8 | 9 | 10 | 11 | 12 13 14 15 16 17
                  |             |         |    |    |           |
               WMI          VDS        Check Year Plant   Sequential
          (Manufacturer) (Descriptor)  Digit            Number

    Valid Characters: 0-9, A-H, J-N, P, R-Z (NO: I, O, Q)


LICENSE
=======

Apache License 2.0 - See LICENSE and NOTICE files

Requires: Attribution, state changes, include license
Permits: Commercial use, modification, distribution, patent use


FOR THE TEAM
============

Quick Setup:

    git clone https://github.com/Thundastormgod/paddleocr_vin_pipeline.git
    cd paddleocr_vin_pipeline
    pip install -r requirements.txt
    pytest tests/test_vin_pipeline.py -v

Data Access:
    The 382 test images are in DagsHub: Thundastormgod/jrl-vin
    Path: data/paddleocr_sample/

Known Limitations:
    - NOT PRODUCTION-READY: 25% vs 95% industry target
    - ENGRAVED PLATES ONLY: Not for printed labels
    - SINGLE VIN PER IMAGE


CITATION
========

If you use this software in research, please cite:

    PaddleOCR VIN Recognition Pipeline
    JRL-VIN Project, 2025
    https://github.com/Thundastormgod/paddleocr_vin_pipeline
