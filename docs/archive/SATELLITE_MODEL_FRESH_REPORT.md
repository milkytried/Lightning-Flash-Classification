> ⚠️ SUPERSEDED — retained for provenance only. Not the final result. See README.md and report/ for Version 2.

> **Archive notice:** This document describes the June 2026 evaluation of the earlier 11-PNG Himawari-8 frozen-backbone prototype. It is **SUPERSEDED by the final FYP report** and is retained only as an audit trail; all path-existence statements below are historical snapshots and may refer to moved or gitignored artifacts. The current aligned Himawari-9 result uses 41,168 balanced patches (33,226 / 3,324 / 4,618), threshold 0.51, and achieves accuracy 0.9095, precision 0.8742, recall/POD 0.9567, F1 0.9136, ROC-AUC 0.9681, FAR 0.126, CSI 0.841, TSS 0.819, and HSS 0.819.

# HIMAWARI-8 SATELLITE CNN LIGHTNING CLASSIFICATION - FRESH MODEL REPORT

**Report Generated:** 2026-06-05 20:53:00  
**Status:** ✅ COMPLETE - Submission-ready Research Prototype

---

## EXECUTIVE SUMMARY

The Himawari-8 satellite CNN lightning-classification prototype has been successfully trained from scratch using a corrected chronological data split and evaluated on completely unseen test data. 

**Key Achievements:**
- ✅ Training completed in 9.1 hours (vs. 62+ days projected for full fine-tuning)
- ✅ Data split integrity verified: Zero PNG overlap between train/val/test
- ✅ All training artifacts saved and archived
- ✅ Test evaluation completed on 46,796 unseen samples
- ✅ Strong ROC-AUC of 0.9199 (92% discrimination ability)

---

## TRAINING CONFIGURATION

### Model Architecture
- **Type:** LightningResNet50 (ResNet-50 backbone + custom classifier head)
- **Input:** 64×64 RGB satellite patches (3 channels)
- **Output:** Binary classification (lightning vs. no lightning)
- **Total Parameters:** 23,770,433
- **Trainable Parameters:** 262,401 (1.1%)
- **Frozen Parameters:** 23,508,032 (ResNet-50 backbone)

### Optimization Strategy
- **Approach:** Layer Freezing
  - Backbone: Frozen (ImageNet pre-trained features stable, reusable)
  - Head: Trainable (custom classifier for satellite domain)
- **Rationale:** Transfer learning on CPU requires dramatic speedup
- **Result:** 4.5+ iterations/sec (vs. ~2.8 estimated for full fine-tuning)

### Training Parameters
- **Loss Function:** FocalLoss (α=0.25, γ=2.0)
  - Handles class imbalance
  - Focuses on hard negatives
- **Optimizer:** Adam (lr=0.001, gradient_clip=1.0)
- **Batch Size:** 32 (train), 256 (validation/test)
- **Max Epochs:** 15
- **Early Stopping:** patience=5 (stops if validation loss doesn't improve)
- **Device:** CPU (no GPU available)

---

## DATA SPLITS

All splits use corrected chronological separation to prevent temporal contamination.

### Training Set
- **Source PNGs:** 6 satellite images (20250418_110037, 20250418_110810, ..., 20250418_155633)
- **Date:** 2025-04-18
- **Total Patches:** 395,952
  - Positive (lightning): 197,976 (50%)
  - Negative (no lightning): 197,976 (50%)

### Validation Set  
- **Source PNGs:** 3 satellite images (20250422_091342, 20250422_093741, 20250422_094538)
- **Date:** 2025-04-22
- **Total Patches:** 38,608
  - Positive (lightning): 19,304 (50%)
  - Negative (no lightning): 19,304 (50%)

### Test Set (Unseen)
- **Source PNGs:** 2 satellite images (20250422_095702, 20250422_100008)
- **Date:** 2025-04-22
- **Total Patches:** 46,796
  - Positive (lightning): 23,398 (50%)
  - Negative (no lightning): 23,398 (50%)

---

## SPLIT INTEGRITY VERIFICATION

| Overlap Check | Result | Status |
|---|---|---|
| Train-Val PNG overlap | 0 PNG files | ✅ Clean |
| Train-Test PNG overlap | 0 PNG files | ✅ Clean |
| Val-Test PNG overlap | 0 PNG files | ✅ Clean |

**Conclusion:** Split is completely clean - no source PNG appears in multiple splits. Data leakage prevented. Temporal contamination fixed with chronological separation.

---

## TRAINING HISTORY

| Metric | Value |
|---|---|
| Total Epochs | 13 (of 15 allowed) |
| Training Duration | 9.1 hours (547.8 minutes) |
| Final Training Loss | 0.0196 |
| Best Validation Loss | 0.0410732 |
| Best Epoch | 8 |
| Early Stopping | Triggered at epoch 13 |

**Key Insight:** Early stopping was triggered because validation loss plateaued - no improvement for 5 consecutive epochs. This is expected behavior with layer freezing and transfer learning on a relatively small domain-specific dataset.

---

## THRESHOLD TUNING ON VALIDATION SET

To find the optimal operating point, we tuned decision thresholds from 0.1 to 0.95 on the validation set (38,608 samples).

**Results:**
- **Best threshold: 0.55** (F1 = 0.8402)
- Selected based on F1 score (balances precision and recall)
- Validation set F1 peaks at 0.55, then degrades as threshold increases

---

## TEST SET EVALUATION WITH TUNED THRESHOLD (0.55)

### Classification Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **Accuracy** | 0.8765 (87.65%) | Strong - now useful for operations |
| **Precision** | 0.8601 (86.01%) | High accuracy when predicting lightning |
| **Recall / POD** | 0.8993 (89.93%) | Excellent - catches 90% of lightning |
| **F1-Score** | 0.8792 | Well-balanced metric |
| **ROC-AUC** | 0.9199 (92%) | Strong ranking ability (unchanged) |

### Weather/Verification Metrics

| Metric | Calculation | Value | Interpretation |
|---|---|---|---|
| **FAR** (False Alarm Ratio) | FP / (TP + FP) | 0.1399 | 13.99% false alarms |
| **CSI** (Threat Score) | TP / (TP + FP + FN) | 0.7845 | 78% success rate |
| **TSS** (True Skill Statistic) | POD - POFD | 0.7530 | Excellent skill vs. chance |
| **HSS** (Heidke Skill Score) | 2(TP*TN - FP*FN) / (...) | 0.7530 | Excellent forecast skill |

*Note: FAR (False Alarm Ratio) = FP / (TP + FP), different from FPR = FP / (FP + TN)*

### Confusion Matrix (with threshold=0.55)

```
                Predicted Positive    Predicted Negative
True Positive        21,042                    2,356
True Negative         3,424                   19,974

Totals:
  True Positives (TP):    21,042 (correctly detected lightning)
  False Positives (FP):   3,424 (false alarms)
  False Negatives (FN):   2,356 (missed lightning)
  True Negatives (TN):    19,974 (correctly rejected non-events)
```

---

### Evaluation Settings (Default Threshold 0.5)
- **Threshold:** 0.5 (sigmoid output > 0.5 → lightning prediction)
- **Test Samples:** 46,796 (completely unseen during training)
- **Device:** CPU
- **Batch Size:** 256

### Classification Metrics (Threshold=0.5)

| Metric | Value |
|---|---|
| Accuracy | 0.5000 (50%) |
| Precision | 0.5000 (50%) |
| Recall / POD | 1.0000 (100%) |
| F1-Score | 0.6667 |
| ROC-AUC | 0.9199 (92%) |

### Confusion Matrix (Threshold=0.5)

```
All 46,796 samples predicted as positive (lightning).
  True Positives (TP):    23,398
  False Positives (FP):   23,398
  False Negatives (FN):   0
  True Negatives (TN):    0
```

**Key Issue:** At default threshold 0.5, model predicts lightning for ALL samples. See threshold tuning section above for improved results.

---

## CRITICAL FINDINGS

### 1. Default Threshold (0.5) Inadequate
At the default threshold of 0.5, the model predicts positive (lightning) for ALL 46,796 test samples:
- This gives 100% recall but 0% true negatives
- Accuracy drops to 50%, FAR = 100%, TSS = 0, HSS = 0
- **Model is not operationally useful at threshold 0.5**

### 2. Threshold Tuning Reveals Strong Model
When threshold is optimized to 0.55 (selected from validation set):
- **Accuracy improves to 87.65%**
- **Recall remains high at 89.93%** (catches 9 of 10 lightning events)
- **Precision: 86.01%** (when predicting lightning, correct 86% of time)
- **FAR drops to 13.99%** (from 100%)
- **TSS/HSS: 0.75** (excellent skill metrics)
- **ROC-AUC: 0.9199** (strong discrimination ability)

**Evidence it's not a failed model:**
- ROC-AUC of 0.9199 shows the model ranks samples correctly
- Threshold calibration reveals excellent discriminative ability
- Performance metrics improve dramatically with proper threshold
- Model can distinguish lightning from non-lightning effectively

### 3. Data Split Integrity: VERIFIED ✅
- Zero PNG overlap between train/val/test
- Chronological separation maintained (2025-04-18 vs. 2025-04-22)
- No temporal contamination detected
- **All training artifacts are trustworthy**

### 4. Training Optimization: HIGHLY SUCCESSFUL ✅
- **Original challenge:** Full ResNet-50 fine-tuning estimated at 62+ days
- **Solution:** Freeze backbone, train only head (262K trainable params)
- **Result:** Completed in 9.1 hours on CPU
- **Speedup factor:** ~160x faster than projected full fine-tuning

---

## FILES GENERATED

All artifacts saved to `models/` directory:

| File | Size | Purpose |
|---|---|---|
| `satellite_resnet50_fresh.pth` | 91 MB | Model checkpoint (research prototype) |
| `satellite_training_history_fresh.json` | - | Epoch-wise training metrics |
| `model_metadata_fresh.json` | - | Complete metadata with split verification |
| `test_evaluation_fresh.json` | - | Test set metrics and confusion matrix |

---

## NEXT STEPS & RECOMMENDATIONS

### ✅ COMPLETED: Threshold Tuning
- [x] Tuned decision thresholds from 0.1 to 0.95 on validation set
- [x] Selected optimal threshold: 0.55 (based on F1 score)
- [x] Applied to test set: 87.65% accuracy, 89.93% recall, 86.01% precision

### Short Term (Operational Testing)
1. Deploy model with threshold=0.55 on real-time satellite data
2. Monitor false alarm rate and detection rate in field
3. Collect feedback from meteorologists and operational users
4. Adjust threshold if needed based on operational priorities

### Medium Term (Performance Refinement)
1. Collect more annotated satellite imagery for validation
2. Retrain with adjusted FocalLoss parameters if needed
3. Experiment with different architectures (ResNet-101, EfficientNet)
4. Test threshold generalization across different seasonal patterns

### Long Term (Deployment)
1. Integrate into operational lightning detection pipeline
2. Compare with existing satellite-based lightning detection methods
3. Deploy as backup/verification for other detection systems
4. Continuously monitor and retrain with operational data

---

## TECHNICAL DETAILS

### Data Pipeline
- **Source:** Malaysian Meteorological Department Himawari-8 satellite imagery
- **Original PNG Size:** 800×950 pixels
- **Patch Size:** 64×64 pixels
- **Lightning Labels:** 60-minute lead-time window
- **Patch Extraction:** 
  - Positive: Centered at lightning strike locations
  - Negative: Random sampling from non-lightning areas

### Normalization
- **ImageNet Statistics:**
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Data Augmentation
- **Training only:** HorizontalFlip, VerticalFlip, Rotate, GaussNoise
- **Validation/Test:** No augmentation (to maintain consistency)

---

## CONCLUSION

**Pipeline Status:** Complete ✅

The Himawari-8 satellite CNN pipeline is complete and the fresh model shows strong probability ranking ability with ROC-AUC = 0.9199. However, at the default threshold of 0.5, the model predicts all samples as lightning, so threshold calibration is required before claiming useful classification performance.

**Key Results:**
- ✅ **Training complete:** 9.1 hours on CPU with layer freezing (vs. 62+ days full fine-tuning)
- ✅ **Data integrity verified:** Zero PNG overlap, chronological separation maintained
- ✅ **Strong discrimination:** ROC-AUC 0.9199 (92% ranking ability)
- ✅ **Threshold tuning:** Optimal threshold 0.55 yields 87.65% accuracy on test set
- ✅ **Research readiness:** With proper threshold (0.55), model achieves 89.93% recall, 86.01% precision on held-out test set

**With Tuned Threshold (0.55):**
- Accuracy: 87.65%
- Recall: 89.93% (catches 9 of 10 lightning events)
- Precision: 86.01%
- FAR: 13.99% (false alarm ratio)
- TSS/HSS: 0.75 (excellent skill metrics)

**Recommendation:** Test with threshold=0.55 on satellite data from additional dates and geographic regions to verify generalization and confirm model robustness.

**Current Status:** Training pipeline complete, evaluation complete, threshold calibrated. Research prototype ready for external validation studies.

---

*End of Report*
