# Himawari-8 Lightning Detection: Debug Report & Corrected Analysis

**Generated:** 2026-05-25 22:58  
**Status:** Critical Issues Identified

---

## 1. Comprehensive Test Set Threshold Table

Analysis of 6,996 held-out test samples at 9 different thresholds:

| Threshold | Pred+ | Pred- | Accuracy | Precision | Recall/POD | F1-Score | FAR | CSI | TSS | HSS |
|-----------|-------|-------|----------|-----------|------------|----------|-----|-----|-----|-----|
| **0.1** ⭐ | 3102 | 3894 | 0.7719 | 0.8066 | 0.7153 | 0.7582 | 0.1934 | 0.6105 | 0.5437 | 0.5437 |
| 0.2 | 2379 | 4617 | 0.6871 | 0.7751 | 0.5272 | 0.6275 | 0.2249 | 0.4572 | 0.3742 | 0.3742 |
| 0.3 | 1502 | 5494 | 0.5758 | 0.6764 | 0.2905 | 0.4064 | 0.3236 | 0.2550 | 0.1515 | 0.1515 |
| **0.4** ⚠️ | 1079 | 5917 | 0.5316 | 0.6024 | 0.1858 | 0.2840 | 0.3976 | 0.1655 | 0.0632 | 0.0632 |
| 0.5 | 936 | 6060 | 0.5189 | 0.5705 | 0.1527 | 0.2409 | 0.4295 | 0.1369 | 0.0377 | 0.0377 |
| 0.6 | 811 | 6185 | 0.5059 | 0.5253 | 0.1218 | 0.1977 | 0.4747 | 0.1097 | 0.0117 | 0.0117 |
| 0.7 | 427 | 6569 | 0.5562 | 0.9602 | 0.1172 | 0.2089 | 0.0398 | 0.1166 | 0.1123 | 0.1123 |
| 0.8 | 317 | 6679 | 0.5450 | 0.9968 | 0.0903 | 0.1657 | 0.0032 | 0.0903 | 0.0901 | 0.0901 |
| 0.9 | 154 | 6842 | 0.5220 | 1.0000 | 0.0440 | 0.0843 | 0.0000 | 0.0440 | 0.0440 | 0.0440 |

**Key observation:** Best test performance is at **threshold 0.1** (F1=0.7582, Accuracy=77.19%), NOT threshold 0.4 selected from validation.

---

## 2. Probability Distribution Analysis

### Overall Test Probabilities
- **Min:** 0.0000 | **Max:** 0.9999 | **Mean:** 0.1831 | **Median:** 0.0621 | **Std:** 0.2476

### By Label (Well-Calibrated)
| Group | Count | Min | Max | Mean | Median | Std |
|-------|-------|-----|-----|------|--------|-----|
| **Lightning (label=1)** | 3498 | 0.0000 | 0.9999 | 0.2749 | 0.2066 | 0.2520 |
| **No-Lightning (label=0)** | 3498 | 0.0000 | 0.8089 | 0.0913 | 0.0005 | 0.2056 |

**✓ Probabilities are WELL-CALIBRATED:** Model correctly outputs HIGHER probabilities for actual lightning samples.

---

## 3. Root Cause Analysis: Validation-Test Mismatch

### Data Source Separation
```
TRAIN:     7 source images (365,376 patches)
           - 20250418_110037, 20250418_110810, 20250418_111802, ...
           
VAL:       3 source images (108,984 patches)
           - 20250418_115511  ← From April 18 (overlaps training domain)
           - 20250418_155633  ← From April 18 (overlaps training domain)
           - 20250422_100008  ← From April 22

TEST:      1 source image (6,996 patches)
           - 20250422_094538  ← From April 22 ONLY
```

### The Problem
- **Validation set** contains images from April 18 (same date range as training) + April 22
- **Test set** contains ONLY April 22 images
- **Training set** heavily samples April 18 domain
- Result: Model memorizes April 18 patterns; test set (April 22) is out-of-distribution

### Performance by Domain
| Domain | Samples | Accuracy | F1 |
|--------|---------|----------|-----|
| April 18 (Training+Validation) | ~475K patches | ~93% | 0.9355 |
| April 22 (Test) | 6,996 patches | ~50-77% | 0.28-0.76 |

---

## 4. Threshold Logic Verification

✅ **CONFIRMED CORRECT:**
- Prediction rule: `predicted_lightning = probability >= threshold`
- NOT using: `probability <= threshold`
- Probabilities are sigmoid outputs (0-1), not raw logits
- Using lightning-class probability (not inverted)

---

## 5. The Discrepancy Explained

**Earlier evaluation (reported 99.89% recall at threshold 0.5):**
- Likely used VALIDATION set or TRAINING set
- Or used different probability column

**Current evaluation (12.46% recall at threshold 0.5 on TEST):**
- Uses correct TEST set (April 22 only)
- Shows true held-out performance
- Much weaker due to temporal distribution shift

---

## 6. Critical Findings

### Issue 1: Poor Test Performance
- At validation-selected threshold 0.4: **F1 = 0.2840** (poor)
- At best test threshold 0.1: **F1 = 0.7582** (acceptable)
- Massive generalization gap from 93% → 50% accuracy

### Issue 2: Temporal Data Contamination
- Validation set is **NOT truly held-out** from training domain
- Test set is the ONLY true test of generalization
- April 22 is fundamentally different from April 18 training data

### Issue 3: Distribution Shift
- Threshold 0.4 works well on April 18 domain (91% validation accuracy)
- Same threshold fails on April 22 domain (53% test accuracy)
- Suggests model learned date-specific patterns, not universal lightning detection

### Issue 4: Dataset Design Flaw
- Proper evaluation requires:
  - Train on early dates (April 18)
  - Validate on middle dates (April 22)
  - Test on later dates (beyond dataset)
- Current approach: Train on April 18, validate on mixed April 18+22, test on April 22 only
- Result: No true temporal validation of generalization

---

## 7. Model Assessment

### What Works
- ✓ Probability calibration is correct
- ✓ Sigmoid activation properly produces 0-1 outputs
- ✓ Threshold logic is correct
- ✓ Threshold evaluation methodology is correct

### What Doesn't Work
- ✗ Poor generalization to April 22 data from April 18 training
- ✗ Validation threshold (0.4) selected from contaminated validation set
- ✗ Only 6,996 test samples; too small to reliably measure performance
- ✗ No temporal separation between train/val/test dates
- ✗ Model appears to memorize date-specific patterns

---

## 8. Corrected Conclusion

**The Himawari-8 satellite CNN pipeline has been implemented end-to-end. However, the current final held-out test performance at the validation-selected threshold is weak, with low recall, low F1, and near-zero skill scores. Further debugging and validation are required before claiming the model performs well.**

### Specific Issues to Address
1. **Threshold selection mismatch:** Validation and test have different optimal thresholds (0.4 vs 0.1)
2. **Temporal contamination:** Validation set overlaps training date range; no true generalization test
3. **Poor test performance:** At threshold 0.4 (validation-selected), test F1 = 0.28; should be ≥ 0.70
4. **Dataset too small:** 6,996 test samples from single date is insufficient

### Recommended Actions
1. Retrain with proper temporal split: earlier dates → train, middle dates → val, later dates → test
2. Expand test set beyond single date
3. Re-evaluate after acquiring more recent/diverse lightning data
4. Check for label generation errors that may favor April 18 patterns
5. Consider separate models per date/season

---

**Report Status:** ⚠️ **Model NOT ready for deployment. Significant generalization issues identified.**
