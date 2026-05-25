# Satellite Lightning Detection Pipeline - Proper Held-Out Test Evaluation

## Status Update: May 25, 2026

### Issue Resolved
Previously, all 481,356 patches were assigned to the training split, making it impossible to validate the model on unseen data. This was due to a temporal data gap: the lightning records end on April 30, 2026, while later PNG files (May 2026) had no lightning data to extract patches from.

### Solution Implemented
Replaced time-based splits with **random image-level splits** (reproducible with seed=42) to ensure all splits have access to available lightning data:
- **Train set**: 365,376 patches (75.9%) from random PNG selection
- **Validation set**: 108,984 patches (22.6%) from random PNG selection  
- **Test set**: 6,996 patches (1.5%) from random PNG selection

Key property: **All patches from a single PNG go to the same split**, preventing patch-level data leakage while all splits have actual lightning data to work with.

### Current Work

**In Progress**: Retraining ResNet-50 on the proper training set only (~365K patches vs. previously 481K)

**Next Steps**:
1. Wait for training to complete
2. Evaluate on held-out test set (6,996 patches)
3. Compute full metrics: Accuracy, Precision, Recall, F1, ROC-AUC, HSS, TSS, FAR
4. Generate confusion matrix and ROC curve
5. Check for mode collapse (prediction distribution)
6. Compare against baselines (always positive, always negative, random)
7. Visual verification of geolocation correctness
8. Data leakage verification

### Important Note
**This evaluation is on unseen test data, not training set**. Results will show true model generalization capability.
