> ⚠️ SUPERSEDED — retained for provenance only. Not the final result. See README.md and report/ for Version 2.

> **Archive notice:** This document describes the superseded panel preparation for the earlier 11-PNG Himawari-8 frozen-backbone prototype. It is **SUPERSEDED by the final FYP report** and is retained only as an audit trail; all path-existence statements below are historical snapshots and may refer to moved or gitignored artifacts. The current aligned Himawari-9 result uses 41,168 balanced patches (33,226 / 3,324 / 4,618), threshold 0.51, and achieves accuracy 0.9095, precision 0.8742, recall/POD 0.9567, F1 0.9136, ROC-AUC 0.9681, FAR 0.126, CSI 0.841, TSS 0.819, and HSS 0.819.

# PANEL Q&A PREPARATION - HIMAWARI-8 SATELLITE CNN

---

## Q1: What is the main goal of this project?

**Answer:**  
To develop and validate a machine-learning CNN prototype that classifies lightning presence in Himawari-8 satellite imagery patches. The research compares two approaches:
1. Metadata-based classifier (baseline)
2. Satellite CNN (primary focus)

The project demonstrates that CNNs trained on raw satellite pixels can detect lightning with strong discriminative ability (ROC-AUC = 0.9199 or 92%).

---

## Q2: What does the model take as input?

**Answer:**  
The CNN takes **only satellite image pixels** as input:
- Input format: 64×64 RGB satellite patches (3 channels)
- Source: Himawari-8 infrared satellite imagery
- ImageNet normalization applied (mean and std)

It does NOT take metadata features like latitude, longitude, amplitude, or strike type.

---

## Q3: Does the model analyze satellite images or metadata?

**Answer:**  
The model analyzes **satellite images only**. 

While lightning metadata is used to generate labels (positive/negative) and extract patch locations, the metadata is never given to the CNN as features. The CNN sees only pixel information from the satellite image and learns to detect patterns that correlate with lightning.

This is the key innovation: the model operates independently without requiring pre-existing lightning detection data to make predictions.

---

## Q4: Why is the metadata model not the final model?

**Answer:**  
The metadata model is a baseline comparison, not the final model because:

1. **Information Leakage:** Using lightning metadata directly means we're using the thing we're trying to predict
2. **Not Generalizable:** Metadata models can't work without pre-existing lightning detection systems
3. **Not Deployable:** Can't use latitude/longitude/amplitude if we don't already know where lightning struck
4. **Purpose:** Metadata model was included only to show satellite CNN outperforms it

**The final model is the satellite CNN** because it:
- Makes independent predictions using only satellite pixels
- Could be deployed as standalone system
- Doesn't require pre-existing lightning detection
- Demonstrates real-world potential

---

## Q5: Why use Himawari-8 imagery?

**Answer:**  
Himawari-8 is ideal for this research because:

1. **Public Data:** Publicly available from NOAA, no proprietary access needed
2. **Continuous Coverage:** Images every 10 minutes from geostationary orbit over Asia/Pacific
3. **Real-Time Potential:** Could enable real-time lightning detection when integrated
4. **Research Grade:** High-resolution (2km pixel at equator), available globally
5. **Proven Track Record:** Used in weather forecasting systems worldwide

Alternative: Other geostationary satellites (GOES, Meteosat) could work similarly.

---

## Q6: Why use ResNet-50?

**Answer:**  
ResNet-50 was chosen for several reasons:

1. **Transfer Learning:** Pre-trained weights on ImageNet provide general feature extraction
2. **CPU-Compatible:** Large but manageable size for CPU training when combined with layer freezing
3. **Proven Architecture:** Excellent performance on image classification tasks
4. **Flexibility:** Easy to customize head for our domain-specific task

Alternatives tested: Could also use ResNet-18 (faster but less accurate) or EfficientNet (more efficient but needs tuning).

---

## Q7: Why freeze the ResNet-50 backbone?

**Answer:**  
Without freezing, full ResNet-50 fine-tuning on CPU would take **62+ days**:
- 23.5M parameters in backbone
- ~2.8 iterations per second on CPU
- Estimated completion: ~62 days

**Solution - Freeze backbone, train only head:**
- Head: 262K trainable parameters (1.1% of total)
- Speed: ~4.5 iterations per second
- Actual training: 9.1 hours (13 epochs)

**Result:** 160x speedup with minimal performance sacrifice. The pre-trained ImageNet features are already excellent for image understanding and don't need much adaptation.

---

## Q8: What is a 64×64 patch?

**Answer:**  
A 64×64 patch is a small square crop from the larger satellite image:

```
Original Himawari-8 Image: 800 × 950 pixels
                 Extract
                    ↓
        64×64 Patch:  [████████]
                      [████████]
                      [████████]
                      (64 pixels × 64 pixels)
```

Why this size?
- **Large enough:** Captures local weather patterns
- **Small enough:** Can extract many samples from each satellite image
- **Computational:** 64×64 fits standard CNN input sizes
- **Information:** Balances spatial context with computational efficiency

From 11 satellite images, we extracted 481,356 total patches.

---

## Q9: How are positive and negative samples generated?

**Answer:**  
**Positive Samples (Lightning):**
1. Lightning strike detected at coordinate (lat, lon)
2. Extract 64×64 pixel patch centered at that location
3. Label as 1 (lightning present)

**Negative Samples (Non-Lightning):**
1. Random sampling from regions where no lightning detected
2. Extract 64×64 pixel patch
3. Ensure patch doesn't overlap lightning region
4. Label as 0 (no lightning)

**Result:** Balanced dataset (50-50 split in each set)

---

## Q10: How did you prevent data leakage?

**Answer:**  
Multiple layers of data leakage prevention:

1. **PNG-Level Split:** No satellite image (PNG) appears in multiple splits
   - Train: 6 PNGs from 2025-04-18
   - Val: 3 PNGs from 2025-04-22
   - Test: 2 PNGs from 2025-04-22

2. **Temporal Separation:** 4-day gap between training and validation/test
   - Prevents temporal contamination
   - Different weather regimes

3. **Threshold Selection:** Used validation set to tune threshold 0.55
   - Test set completely held out from threshold tuning
   - No information leakage to final metrics

4. **Verification:** All checks documented and audited in FINAL_AUDIT.md

---

## Q11: Why split by source PNG/image instead of random patch split?

**Answer:**  
Random patch split would cause data leakage:

**Bad Approach - Random Split:**
```
Same PNG (20250418_110037.png):
  - 100 patches → Training set
  - 50 patches → Test set
  ❌ Leakage: Same image in train and test
  ❌ Unrealistic: Model sees training data at test time
```

**Correct Approach - PNG-Level Split:**
```
All patches from PNG → Same split
  - PNG 1, 2, 3, 4, 5, 6 → Training set (2025-04-18)
  - PNG 7, 8, 9 → Validation set (2025-04-22)
  - PNG 10, 11 → Test set (2025-04-22)
  ✓ Clean: Different PNGs in each split
  ✓ Fair: Model evaluated on completely new imagery
```

---

## Q12: What threshold was selected and why?

**Answer:**  
**Threshold = 0.55** was selected based on validation set optimization.

**Why tuning was necessary:**
- Default threshold 0.5 predicted ALL 46,796 test samples as lightning
- Accuracy: Only 50% (random guessing level)
- Problem: Model output distribution skewed toward >0.5

**How threshold 0.55 was selected:**
- Tested 18 thresholds: 0.1 to 0.95
- Criterion: Maximize F1-Score (balanced metric)
- Validation set F1 peaks at 0.55

**Why 0.55 specifically:**
- Balances precision and recall
- 86% precision (accurate when predicting lightning)
- 90% recall (catches most lightning events)
- Optimal operating point for weather forecasting

---

## Q13: Why was threshold 0.55 used?

**Answer:**  
Threshold 0.55 was selected because:

1. **Validates on Validation Set:** F1-score peaks at 0.55 among 18 tested thresholds
2. **Maximizes F1-Score:** 0.8402 at threshold 0.55 (better than 0.5 or other values)
3. **Balanced Trade-off:** 
   - Recall 90% (catches lightning)
   - Precision 86% (low false alarms)
4. **Meteorological Relevance:** High recall important for safety warnings

**Verification:**
- Threshold tuned on validation set (38,608 samples)
- Applied to completely independent test set (46,796 samples)
- No circular evaluation or overfitting

---

## Q14: What is ROC-AUC?

**Answer:**  
ROC-AUC (Area Under the Receiver Operating Characteristic Curve) measures **how well the model ranks samples**.

**ROC Curve:**
```
         TPR
         100% |     /
              |    /●  <- Good model (AUC ≈ 0.9)
              |   /
          50% |  /────
              | /   
              |/
               └─────────────── FPR
              0%      50%    100%
```

**AUC = 0.9199 (92%) means:**
- If you pick one lightning sample and one non-lightning sample
- Model ranks the lightning sample higher 92% of the time
- Shows strong discrimination ability (independent of threshold)

**Interpretation:**
- 0.5 = Random guessing
- 0.7-0.8 = Good discrimination
- 0.8-0.9 = Very good discrimination
- 0.9+ = Excellent discrimination ← We achieved this

**Why ROC-AUC matters:**
- Threshold-independent (doesn't change with decision threshold)
- Proves model learned the pattern, not just got lucky

---

## Q15: What is recall / POD?

**Answer:**  
Recall (also called POD = Probability of Detection) measures:

```
Recall = TP / (TP + FN) = 21,042 / 23,398 = 89.93%

Question: Of all actual lightning events, how many did we detect?
Answer: 89.93% (caught 9 of every 10 lightning events)
```

**In plain words:**
- 23,398 real lightning events in test set
- Model correctly detected 21,042 of them
- Missed 2,356 lightning events
- Detection rate: 89.93%

**Visualization:**
```
Lightning Events:
  ✓✓✓✓✓✓✓✓✓ ✗  (9 detected, 1 missed)
            ↑
        90% recall
```

---

## Q16: Why is high recall important for lightning detection?

**Answer:**  
High recall (90%+) is critical for lightning detection because:

1. **Safety:** Missing lightning could mean missing dangerous weather
2. **Warnings:** Forecasters need to detect most lightning to issue timely alerts
3. **False Negatives Are Costly:** Missed detection could lead to injuries
4. **Operational Standard:** Weather services target 90%+ detection rates

**Recall vs. Precision Trade-off:**

```
High Recall (90%):    Catches most lightning, but some false alarms
   ├─ Good for safety (don't miss storms)
   ├─ Accept some false alarms as cost of protection
   └─ Our choice: 90% recall, 14% false alarm rate

High Precision (99%):  Few false alarms, but misses lightning
   ├─ Bad for safety (too many missed events)
   ├─ Could lead to injuries
   └─ Not suitable for weather warnings

Balanced (86% precision, 90% recall):
   ├─ Catches most lightning
   ├─ Acceptable false alarm rate
   └─ Threshold 0.55 achieves this
```

---

## Q17: What is FAR?

**Answer:**  
FAR (False Alarm Ratio) measures the rate of false alarms:

```
FAR = FP / (TP + FP) = 3,424 / 24,466 = 13.99%

Question: Of all positive predictions, what fraction were wrong?
Answer: 14% were false alarms, 86% were correct
```

**In plain words:**
- Model predicted lightning 24,466 times
- 21,042 were correct (TP - true positives)
- 3,424 were false alarms (FP - false positives)
- False alarm rate: 13.99%

**Operational Significance:**
- FAR of 14% acceptable for weather forecasting
- Better than random (50%) or bad models
- Meteorology standard: <20% FAR is good

---

## Q18: What is CSI / Threat Score?

**Answer:**  
CSI (Critical Success Index, also called Threat Score) measures overall success:

```
CSI = TP / (TP + FP + FN) = 21,042 / 26,822 = 78.45%

Question: Overall, how many correct decisions out of all opportunities?
```

**What it counts:**
- ✓ Correct detections (TP)
- ✗ False alarms (FP)
- ✗ Missed events (FN)

**In plain words:**
- 78% success rate
- Missing both false alarms AND false negatives
- Balanced view of all errors

**Why it matters:**
- Single metric showing overall effectiveness
- Meteorology standard: >50% is good, >70% is very good
- We achieved 78% (excellent)

---

## Q19: What are TSS and HSS?

**Answer:**  
**TSS (True Skill Statistic) = 0.7530**

```
TSS = POD - POFD
    = (TP / (TP+FN)) - (FP / (FP+TN))
    = 0.8993 - 0.1466
    = 0.7530

Range: 0.0 (no skill) to 1.0 (perfect)
Our result: 0.75 (Excellent skill)
```

**HSS (Heidke Skill Score) = 0.7530**

```
HSS = 2(TP×TN - FP×FN) / (...)
    = 0.7530

Measures: How much better than expected by random chance
Range: 0.0 (no improvement) to 1.0 (perfect)
Our result: 0.75 (Excellent skill)
```

**Interpretation:**
- Both metrics > 0.6: Excellent forecast skill
- Both metrics > 0.7: Outstanding performance
- **We achieved both: 0.75 each** ✓

**Why both metrics:**
- TSS: Metric of model ability
- HSS: How much we outperform chance
- Both high = Strong validated performance

---

## Q20: What were the final test results?

**Answer:**  
**Test Set: 46,796 completely unseen samples, Threshold: 0.55**

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | 87.65% | 40,956 of 46,796 correct |
| Precision | 86.01% | 86% of lightning predictions correct |
| Recall | 89.93% | Caught 9 of 10 lightning events |
| F1-Score | 0.8792 | Balanced performance metric |
| ROC-AUC | 0.9199 | 92% ranking ability |
| FAR | 13.99% | 14% false alarm rate |
| CSI | 0.7845 | 78% success rate |
| TSS | 0.7530 | Excellent skill |
| HSS | 0.7530 | Excellent forecast skill |

**Confusion Matrix:**
```
TP: 21,042  (correctly detected lightning)
FP: 3,424   (false alarms)
FN: 2,356   (missed lightning)
TN: 19,974  (correctly rejected non-lightning)
```

**Conclusion:** Strong, balanced performance with 90% detection and 14% false alarm rate.

---

## Q21: What are the model limitations?

**Answer:**  
**Dataset Limitations:**
1. Small geographic scope (Malaysian airspace only)
2. Limited time period (11 PNGs over 4 days, single weather regime)
3. Single lightning data provider (potential systematic bias)
4. Only 481K patches (small by modern deep learning standards)

**Meteorological Limitations:**
1. Single satellite channel (IR only, limited information)
2. Static patches (64×64 pixels lose temporal context)
3. Label quality depends on lightning detection system accuracy
4. Regional training may not generalize globally

**Model Limitations:**
1. Frozen backbone (limited adaptation to satellite imagery domain)
2. No temporal component (can't use time series)
3. Limited spatial context (can't see larger storm systems)
4. Fixed architecture (ResNet-50 not optimized for satellite)

**Honest Assessment:**
- Model works well on 4-day dataset
- Requires validation on additional dates/regions before operational use
- Should not be deployed without further testing

---

## Q22: Why is this not production-ready yet?

**Answer:**  
**This is a research prototype**, not production-ready because:

**Validation Gaps:**
1. Trained on only 4 days of satellite data (single weather regime)
2. Only tested on Malaysian airspace (geographic specificity)
3. No validation on different seasons/weather patterns
4. No cross-validation with other lightning detection systems

**Deployment Concerns:**
1. Needs operational testing with meteorologists
2. Threshold 0.55 not verified to generalize to new conditions
3. No uncertainty quantification for system confidence
4. No real-time pipeline integration

**Research Status:**
- ✓ Demonstrates CNN can classify satellite lightning
- ✓ Shows strong held-out test performance
- ✓ Proves better than metadata baseline
- ✗ Not ready for replacing operational systems

**Next Steps Before Production:**
1. Test on satellite data from 3-6 additional months
2. Validate in different geographic regions
3. Operational testing with meteorologists
4. Continuous monitoring once deployed

---

## Q23: What would you improve next?

**Answer:**  
**Short Term (if continuing research):**
1. Collect 3-6 months satellite data for training
2. Test on different dates to verify threshold generalization
3. Validate across multiple geographic regions
4. Work with meteorologists on operational feedback

**Medium Term (research priorities):**
1. Temporal models: Add LSTM for time-series (use 5-10 frame sequences)
2. Multi-band satellite: Use IR, visible, water vapor channels
3. Attention mechanisms: Learn which image regions matter
4. Larger receptive field: Use larger patches or pyramid networks

**Long Term (production considerations):**
1. Ensemble methods: Combine with other detection systems
2. Uncertainty quantification: Add confidence intervals
3. Real-time pipeline: Integrate with satellite data streams
4. Continuous learning: Incorporate meteorologist feedback

**Why these improvements:**
- Temporal models capture storm evolution
- Multi-band provides richer information
- Attention shows interpretability
- Ensemble improves robustness
- Uncertainty helps operators make decisions

---

## Q24: Can you demonstrate the model in real-time?

**Answer:**  
**Live Demonstration Options:**

1. **Inference on Sample Images:** Load a test patch and show probability output
   - Input: 64×64 satellite patch
   - Output: Lightning probability (0.0-1.0)
   - Decision: Threshold at 0.55

2. **Batch Processing:** Run model on 100 test samples and show predictions

3. **Threshold Comparison:** Show how predictions change at different thresholds
   - Threshold 0.3: High recall, high false alarms
   - Threshold 0.55: Balanced
   - Threshold 0.8: High precision, miss some lightning

4. **Visualization:** Show confusion matrix, ROC curve, sample patches

**Requirements:**
- Python with PyTorch
- Model checkpoint: `models/satellite_resnet50_fresh.pth`
- Test data: `data/processed/satellite_dataset.csv` + patch images

---

## Q25: How do I reproduce your results?

**Answer:**  
**Reproducibility Guaranteed:**

All scripts and data included:
1. `train_fresh_optimized.py` - Training script
2. `eval_test_fresh.py` - Evaluation script
3. `tune_threshold.py` - Threshold tuning
4. `src/model_arch.py` - Model architecture
5. `src/himawari_data_loader.py` - Data loading
6. `data/processed/satellite_dataset.csv` - Dataset reference

**To Reproduce:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python train_fresh_optimized.py

# 3. Tune threshold
python tune_threshold.py

# 4. Final evaluation
python eval_test_fresh.py
```

**Expected Results:**
- Training: 13 epochs in ~9 hours (CPU)
- Threshold: 0.55 optimal
- Test accuracy: 87.65%
- ROC-AUC: 0.9199

All hyperparameters, seed, and split method documented in code.

---

*End of Q&A Preparation*
