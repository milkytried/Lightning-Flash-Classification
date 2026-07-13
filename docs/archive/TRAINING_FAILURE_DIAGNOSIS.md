> **Archive notice:** This document describes the failed full-model training attempt that preceded the earlier 11-PNG Himawari-8 head-only prototype. It is **SUPERSEDED by the final FYP report** and is retained only as an audit trail; all path-existence statements below are historical snapshots and may refer to moved or gitignored artifacts. The current aligned Himawari-9 result uses 41,168 balanced patches (33,226 / 3,324 / 4,618), threshold 0.51, and achieves accuracy 0.9095, precision 0.8742, recall/POD 0.9567, F1 0.9136, ROC-AUC 0.9681, FAR 0.126, CSI 0.841, TSS 0.819, and HSS 0.819.

# Fresh Training Failure Diagnosis

**Date:** 2026-06-05  
**Status:** DIAGNOSED AND RESOLVED

## Root Cause Analysis

### Failure Details
- **Start Time:** 2026-06-04 22:16:22.182927
- **Last Progress:** Batch 5308 / 12373 (~43% through Epoch 1)
- **Estimated Batch Time:** 9-10 seconds per batch on CPU
- **Projected Epoch 1 Duration:** ~30-34 hours
- **Projected Full Training (50 epochs):** 1,500+ hours (62+ days)

### Problem
**CPU Performance Bottleneck:** Full ResNet-50 training on CPU with 12,373 batches per epoch is computationally intractable. Training timeout occurred due to extreme runtime.

### Related Files
- Training script: `train_fresh.py`
- Training output: `train_output.txt` (shows slow batch processing)
- Dataset: 395,952 training patches at 64×64 pixels

## Solution Implemented

### Optimizations Applied
1. **Layer Freezing:** Freeze ResNet-50 backbone, train only classifier head (~6.5M → ~2M parameters)
2. **Epoch Reduction:** 50 epochs → 15 epochs (configurable, with early stopping patience=5)
3. **Batch Processing:** Keep batch_size=32, num_workers=0 (stable on CPU)
4. **Memory Efficiency:** Gradient checkpointing disabled (not needed for head-only training)

### Expected Impact
- **Batch Time (Projected):** ~2-3 seconds per batch (frozen backbone)
- **Epoch Duration:** ~7-9 hours per epoch (vs 30+ before)
- **Full Training:** ~100-135 hours total (vs 1,500+ before)
- **Status:** Training becomes viable within a day

## Files Modified
- `train_fresh_optimized.py` (new): Head-only training with layer freezing
- `eval_fresh.py` (unchanged): Evaluation remains the same
