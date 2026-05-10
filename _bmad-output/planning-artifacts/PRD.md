# Product Requirements Document
## Lightning Flash Classification Using Himawari-8 Satellite Imagery

**Project Name:** Deep Learning for Lightning Flash Classification Using Geostationary Satellite Imagery  
**Author:** Bryan Chai Wen Cheng (23073679)  
**Supervisor:** Associate Professor Ir Ts. Dr Wong Shen Yuong  
**Status:** DRAFT — PRD v1.0  
**Created:** 2026-05-10  
**Target Completion:** 2026-08-31  

---

## 1. Executive Summary

This capstone project will develop a CNN-based deep learning model to predict cloud-to-ground (CG) lightning occurrence over Malaysia within a 0–60 minute forecast window using Himawari-8 geostationary satellite imagery and historical ground lightning records from the Malaysian Meteorological Department (MMD).

**Primary Deliverable:** A reproducible, well-documented CNN model capable of nowcasting lightning with >85% recall and <30% false alarm ratio, suitable for academic evaluation and potential MMD integration.

**Success Definition:** Model meets all success criteria (Section 3), reproducible preprocessing pipeline, comprehensive documentation, and supervisor approval.

---

## 2. Product Vision & User Personas

### 2.1 Primary Customer: Academic Evaluation
**Who:** Capstone Supervisor (Dr. Wong) + Assessment Panel  
**What They Care About:**
- Rigorous methodology grounded in literature
- Reproducible, well-documented code
- Clear evaluation metrics and error analysis
- Proper risk acknowledgment and limitations
- Realistic timeline and achievable scope

**Success for Them:** A-grade capstone demonstrating deep learning competency in applied geophysics context.

### 2.2 Secondary Customer: MMD & Meteorological Community
**Who:** Malaysian Meteorological Department, meteorological analysts, domain researchers  
**What They Care About:**
- Practical utility and lead time (10–30 min minimum)
- False alarm rate acceptable to operational use
- Generalization to unseen weather patterns
- Integration feasibility with existing systems

**Success for Them:** Proof-of-concept that satellite-only CG lightning nowcasting is viable over Malaysia; usable benchmark for future operational development.

### 2.3 Tertiary Stakeholder: Safety Community
**Who:** Event organizers, aviation, infrastructure operators  
**What They Care About:** Advance warning of lightning risk  
**Success for Them:** Early prototype that could eventually feed into public warning systems (post-capstone).

---

## 3. Success Criteria & Acceptance Thresholds

### 3.1 Model Performance Metrics

| Category | Metric | Target | Rationale |
|----------|--------|--------|-----------|
| **Detection** | Recall (POD) | ≥85% | Minimize missed lightning (safety critical) |
| **False Alarm** | False Alarm Ratio (FAR) | <30% | Keep model useful; avoid warning fatigue |
| **Overall Skill** | ROC-AUC | ≥0.85 | Strong discrimination between lightning/no-lightning |
| **Balance** | F1-Score | ≥0.70 | Reflects recall/precision trade-off |
| **Meteorological** | Heidke Skill Score (HSS) | ≥0.40 | Skill vs. climatology baseline |

**Optimization Priority:** Recall > Precision (minimize false negatives due to safety implications).

### 3.2 Lead Time & Operational Requirements

| Requirement | Target | Acceptance |
|-------------|--------|-----------|
| **Minimum Lead Time** | 10–30 minutes | ≥10 min (acceptable); 20–30 min preferred |
| **Inference Speed** | <5 sec per image | Per-image latency <5 sec on GPU |
| **Dataset Coverage** | 2018–2020 MMD data | ≥2 years of continuous records |
| **Spatial Accuracy** | 2 km/pixel (IR) | Acceptable; aligns with Himawari-8 AHI native resolution |

### 3.3 Deliverables Acceptance Criteria

| Deliverable | Acceptance Criteria |
|-------------|-------------------|
| **Dataset** | Preprocessed, balanced, labeled; reproducible build scripts; ≥10k labeled patches or images |
| **Baseline Model** | ResNet-50 classifier trained; weights saved; reproducible training logs |
| **Evaluation Report** | All metrics computed; error analysis with visual examples; comparison table (variants) |
| **Code Repository** | Git tracked; well-commented; reproducible environment (requirements.txt/conda); README |
| **Documentation** | Methods paper (5–10 pages); API docs; deployment guide; limitations & future work |
| **Presentation** | Supervisor-ready slides + demo (live inference on test set) |

### 3.4 Scope Constraints (Hard Limits)

- **No long-range forecasting** (>60 min lead time out of scope)
- **No intra-cloud lightning classification**
- **No multi-source fusion** (radar, weather stations, NWP models)
- **No real-time operational deployment** (prototype only)
- **No public warning system integration** (documentation only)

---

## 4. Requirements Breakdown

### 4.1 Functional Requirements

#### FR1: Data Acquisition & Integration
- **FR1.1** Acquire MMD Lightning Detection System records (2018–2020) with time, lat/lon, CG strike type
- **FR1.2** Acquire Himawari-8 AHI imagery (bands: 10.4 µm IR, 6.2 µm water vapor, 0.64 µm visible)
- **FR1.3** Crop imagery to regional box (20°×20° covering Peninsular + East Malaysia)
- **FR1.4** Validate temporal alignment (±10 min tolerance for image-lightning matching)

#### FR2: Data Preprocessing & Labeling
- **FR2.1** Reproject lightning lat/lon to Himawari-8 pixel coordinates (handle parallax 1–2 px)
- **FR2.2** Cloud masking: exclude pixels with brightness temp >290 K (clear sky / shallow clouds)
- **FR2.3** Normalize each band to [0, 1]
- **FR2.4** Stack multi-channel tensors (3–5 bands)
- **FR2.5** Label patches: `1` if ≥1 CG strike in [t, t+60min] within spatial area, else `0`
- **FR2.6** Downsample negatives or apply class-weighted loss for balance
- **FR2.7** Augment training data (flips, rotations, intensity jitter)

#### FR3: Model Training
- **FR3.1** Implement ResNet-50 patch classifier with ImageNet pretraining (adapt first conv for N channels)
- **FR3.2** Binary cross-entropy or Focal Loss; Adam/SGD optimizer; learning rate scheduling
- **FR3.3** Train on 2018–2019 data; validate on held-out 2019; test on 2020 (time-ordered splits)
- **FR3.4** Early stopping on validation loss; save best weights
- **FR3.5** (Optional) Implement U-Net segmentation for comparison

#### FR4: Model Evaluation
- **FR4.1** Compute accuracy, precision, recall, F1, ROC-AUC on test set
- **FR4.2** Compute meteorological metrics: POD, FAR, HSS, TSS
- **FR4.3** Error analysis: visualize false positives & false negatives; inspect satellite patches
- **FR4.4** Comparison table: baseline vs. optional variants (single band vs. multi-band, ResNet vs. U-Net)

#### FR5: Documentation & Deployment
- **FR5.1** Methods paper: data, preprocessing, model, evaluation
- **FR5.2** API documentation: inference function signature, input/output format
- **FR5.3** Deployment guide: Docker setup (optional), inference script, GPU requirements
- **FR5.4** User manual: limitations, when to trust/distrust model, integration notes for MMD

### 4.2 Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| **Reproducibility** | Code + data build scripts enable full re-execution on fresh environment |
| **Performance** | Inference <5 sec/image on NVIDIA GPU (≥8 GB VRAM) |
| **Scalability** | Batch processing ≥32 patches (classifier) or 1–2 images (U-Net) |
| **Code Quality** | Well-commented; type hints; unit tests for preprocessing |
| **Version Control** | Git tracked with meaningful commits; `.gitignore` for data & large weights |
| **Environment** | Python 3.9+; TensorFlow/Keras or PyTorch; documented dependency versions |

---

## 5. Implementation Roadmap & Milestones

### Phase 1: Foundation & Data (Weeks 1–4)
**Goal:** Secure data; establish development environment; validate alignment.

| Milestone | Deliverable | Success Criteria |
|-----------|-------------|-----------------|
| **M1.1** Data Acquisition Complete | MMD records (2018–2020); Himawari-8 archive access confirmed | ≥2 years data; both sources accessible |
| **M1.2** Dev Environment Setup | Git repo; Python env; GPU access verified; data folder structure | Code runs on supervisor's machine |
| **M1.3** Alignment Validation | Time/spatial alignment confirmed; sample imagery-lightning matched | ±10 min temporal tolerance; <2 px spatial error |

**Dependency:** MMD data approval from institution (critical path). If delayed >2 weeks, activate contingency (Section 5.2).

---

### Phase 2: Preprocessing Pipeline (Weeks 5–8)
**Goal:** Build reproducible dataset from raw data; validate labels.

| Milestone | Deliverable | Success Criteria |
|-----------|-------------|-----------------|
| **M2.1** Preprocessing Scripts | Cropping, reprojection, cloud masking, normalization | Runs on raw data; outputs consistent tensors |
| **M2.2** Labeling Pipeline | Patch/pixel labeling; class balance analysis; augmentation | ≥10k labeled patches; imbalance quantified |
| **M2.3** Dataset Splits | Train (2018–2019), Val (2019 holdout), Test (2020) | No data leakage; time-ordered splits validated |
| **M2.4** QA Report | Visual inspection; statistics (distribution, missing data) | No NaNs; pixels in [0,1]; labels balanced |

---

### Phase 3: Model Development (Weeks 9–14)
**Goal:** Train baseline & optional models; tune hyperparameters.

| Milestone | Deliverable | Success Criteria |
|-----------|-------------|-----------------|
| **M3.1** Baseline Model (ResNet-50) | Model architecture; training loop; logging | Trains without errors; loss decreasing |
| **M3.2** Hyperparameter Tuning | Learning rate sweep; batch size optimization; loss function comparison | Converges to stable validation loss |
| **M3.3** Best Model Selection | Weights saved; threshold tuned via ROC curve | Validation ROC-AUC ≥0.82 |
| **M3.4** (Optional) U-Net Model | Segmentation architecture; training on full images | Comparison table generated |

---

### Phase 4: Evaluation & Analysis (Weeks 15–18)
**Goal:** Comprehensive test evaluation; error deep-dive.

| Milestone | Deliverable | Success Criteria |
|-----------|-------------|-----------------|
| **M4.1** Test Set Evaluation | All metrics computed; classification report | Recall ≥85%; FAR <30%; ROC-AUC ≥0.85 |
| **M4.2** Error Analysis | FP/FN examples; satellite patch visualization; confusion matrix | Visual insights into failure modes |
| **M4.3** Meteorological Validation | POD, FAR, HSS, TSS computed; skill assessment | HSS ≥0.40; leads interpreted meteorologically |
| **M4.4** Comparison Report | ResNet vs. optional U-Net; single vs. multi-band | Clear recommendation on best variant |

---

### Phase 5: Documentation & Delivery (Weeks 19–22)
**Goal:** Finalize all deliverables; prepare presentation.

| Milestone | Deliverable | Success Criteria |
|-----------|-------------|-----------------|
| **M5.1** Methods Paper | 5–10 pages; lit review, methods, results, discussion | Supervisor review complete; revisions incorporated |
| **M5.2** Code Documentation | API docs; README; deployment guide; limitations | Code runs end-to-end from README |
| **M5.3** Final Report & Slides | Capstone report; presentation slides; live demo script | Ready for supervisor + assessment panel |
| **M5.4** Repository Submission | Clean Git history; final tagged release | All deliverables in `_bmad-output/` |

**Total Timeline:** 22 weeks (Sept 2025 – Jan 2026, adjusted to actual capstone cycle).

---

## 5.1 Weekly Checkpoints (Agile Cadence)

- **Weekly Sync (Every Monday):** Progress update, blockers, next week priorities
- **Bi-weekly Metrics Review:** Training curves, dataset statistics, emerging risks
- **Milestone Gate Reviews:** Supervisor approval before advancing to next phase

---

## 5.2 Risk Mitigation & Contingencies

### Top Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| **MMD data delayed** | HIGH | CRITICAL | Contingency: Use synthetic labels from public rainfall radar or adjacent country's GLM data; slower progress but unblocks model development |
| **Class imbalance too severe** | MEDIUM | HIGH | Focal Loss + aggressive negative downsampling; oversampling minority class |
| **GPU memory overflow** | MEDIUM | MEDIUM | Reduce batch size to 16; use gradient checkpointing; consider smaller ResNet variant (ResNet-18) |
| **Geolocation parallax errors** | MEDIUM | MEDIUM | Tolerance bands ±2 px; visual validation of sample alignments; document tolerance in report |
| **Model overfits to 2018–2019 patterns** | MEDIUM | HIGH | Regularization (dropout, L2); aggressive augmentation; early stopping; 2020 test set is truly held-out |
| **Insufficient lead time achieved** | MEDIUM | MEDIUM | Extend lead-time window to 0–90 min (trade-off: more false alarms); document in limitations |

**Fallback Plan if critical delays occur:**
- Reduce to single ResNet-50 baseline (skip U-Net comparison)
- Use public Himawari-8 archive with synthetic labels
- Compress documentation; focus on methods reproducibility
- Still deliver functional model + honest limitations assessment

---

## 6. Metrics & Evaluation Strategy

### 6.1 Dataset Metrics
- **Total samples:** ≥10k labeled patches or images
- **Class balance:** Lightning:Non-lightning ratio quantified (typically 1:10 to 1:20)
- **Spatial coverage:** Full Malaysia + neighboring regions (20°×20° box)
- **Temporal range:** 2018–2020 continuous (≥3 years)

### 6.2 Model Performance Metrics (on Test Set 2020)

```python
# Core ML Metrics
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall (POD) = TP / (TP + FN)       # ← PRIMARY METRIC (target ≥85%)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
ROC-AUC = Area Under ROC Curve       # ← (target ≥0.85)

# Meteorological Metrics
Probability of Detection (POD) = Recall
False Alarm Ratio (FAR) = FP / (TP + FP)  # ← (target <30%)
Heidke Skill Score (HSS) = (Po - Pe) / (1 - Pe)  # ← (target ≥0.40)
True Skill Statistic (TSS) = POD - FAR
```

### 6.3 Error Analysis & Reporting
- **Confusion matrix:** Breakdown of TP, TN, FP, FN
- **Visual examples:** 20–30 misclassified patches with satellite images
- **ROC curve:** Threshold sweep; recommended operating point
- **Lead time distribution:** Histogram of actual lead times achieved
- **Seasonal breakdown:** Performance by monsoon season (if data permits)

---

## 7. Technical Architecture & Tools

### 7.1 Tech Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| **Language** | Python 3.9+ | Industry standard for ML; TensorFlow/PyTorch native |
| **DL Framework** | TensorFlow/Keras (primary); PyTorch (optional) | Keras easier for rapid prototyping; PyTorch for comparison |
| **Data** | NumPy, pandas, xarray/netCDF4 | Satellite data in netCDF; lightning in CSV |
| **Geospatial** | Cartopy, PyProj | Reprojection; coordinate transforms; mapping |
| **Experiment** | TensorBoard; MLflow (optional) | Training visualization; hyperparameter tracking |
| **Testing** | pytest; unittest | Unit tests for preprocessing; integration tests |
| **Version Control** | Git + GitHub/GitLab | Reproducible history; code review |
| **Environment** | Conda or venv | Isolated dependencies; reproducibility |
| **Hardware** | NVIDIA GPU ≥8 GB VRAM | ResNet-50 batch 32 ≈ 6 GB; margin for overhead |

### 7.2 Repository Structure

```
Project-Capstone/
├── data/
│   ├── raw/                      # MMD CSV + Himawari-8 netCDF (gitignored)
│   ├── processed/                # Cropped imagery, labeled patches
│   └── splits/                   # Train/val/test indices
├── src/
│   ├── preprocessing.py          # Cropping, labeling, augmentation
│   ├── model_arch.py             # ResNet-50, U-Net definitions
│   ├── train.py                  # Training loop, checkpointing
│   ├── evaluate.py               # Metrics computation, error analysis
│   └── inference.py              # Per-image prediction API
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing_viz.ipynb # Labeling validation
│   └── 03_results.ipynb          # Final visualizations
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── docs/
│   ├── METHODS.md                # Methods paper
│   ├── API.md                    # API documentation
│   ├── DEPLOYMENT.md             # Deployment guide
│   └── LIMITATIONS.md            # Honest limitations & future work
├── models/
│   └── best_resnet50.h5          # Saved weights (gitignored)
├── results/
│   ├── metrics.json              # Test metrics
│   ├── confusion_matrix.png
│   └── error_examples/           # FP/FN visualizations
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── README.md                     # Quick start guide
└── .gitignore                    # Excludes: data/, models/, __pycache__/
```

---

## 8. Success Criteria Summary & Go/No-Go Checklist

### Final Acceptance Gates

- [ ] **Model Performance:** Recall ≥85%, FAR <30%, ROC-AUC ≥0.85 on held-out 2020 test set
- [ ] **Dataset:** ≥10k labeled samples; no data leakage; reproducible build pipeline
- [ ] **Code Quality:** All code in Git; well-commented; reproducible environment (requirements.txt)
- [ ] **Documentation:** Methods paper, API docs, deployment guide, limitations clearly stated
- [ ] **Evaluation Report:** All metrics, confusion matrix, error analysis, visual examples
- [ ] **Reproducibility:** End-to-end pipeline executable from README; supervisor can rerun
- [ ] **Presentation:** Slides ready; live demo executable; supervisor sign-off

**Capstone Success Outcome:** A-grade assessment; potential MMD interest in prototype; publication-ready baseline for future research.

---

## 9. Assumptions & Dependencies

### Critical Assumptions
1. **MMD data is accessible** within 2 weeks of project start (institutional approval)
2. **GPU with ≥8 GB VRAM available** for training (supervisor's lab or cloud compute)
3. **Himawari-8 archive (JMA/NOAA) remains publicly available** and stable
4. **No significant sensor/data quality changes** across 2018–2020 period

### External Dependencies
- MMD Lightning Detection System (data source)
- JMA / NOAA Himawari-8 archive (satellite imagery)
- TensorFlow/Keras or PyTorch ecosystem (stable releases)
- NVIDIA CUDA/cuDNN (GPU acceleration)

### Internal Dependencies
- Supervisor technical feedback (weekly)
- Computing resources (GPU access)
- Institutional data governance approval (if applicable)

---

## 10. Stakeholder Communication Plan

| Stakeholder | Frequency | Channel | Content |
|-------------|-----------|---------|---------|
| **Supervisor (Dr. Wong)** | Weekly | In-person/Zoom | Progress, blockers, technical decisions |
| **Assessment Panel** | Mid-project + Final | Presentation | Interim report (Week 10); Final presentation (Week 22) |
| **MMD (Future)** | Post-capstone | Email + Report | Offer prototype + documentation for potential collaboration |

---

## 11. Glossary & Definitions

| Term | Definition |
|------|-----------|
| **CG Lightning** | Cloud-to-ground lightning strike; electrical discharge from cloud to ground |
| **Nowcasting** | 0–60 minute forecasting (very short-term predictions) |
| **Lead Time** | Time between prediction and actual event occurrence (goal: 10–30 min) |
| **POD** | Probability of Detection (Recall); % of actual lightning events correctly predicted |
| **FAR** | False Alarm Ratio; % of predicted lightning that did not occur |
| **HSS** | Heidke Skill Score; skill vs. climatology baseline |
| **Patch** | 64×64 pixel subsection of satellite image (~128×128 km) |
| **Himawari-8 AHI** | Advanced Himawari Imager; 16-channel multispectral sensor on Himawari-8 satellite |
| **Transfer Learning** | Reusing weights from ImageNet-pretrained ResNet-50; adapting to new task |
| **Class Imbalance** | Lightning events rare (~5–10% of dataset); requires special handling |

---

## 12. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-10 | John (PM) + Bryan | Initial PRD from proposal & brief analysis |

---

## 13. Appendices

### A. Proposed Model Variants for Comparison

**Baseline (Primary):**
- ResNet-50 + transfer learning
- 64×64 patch classifier
- Binary cross-entropy loss
- Expected: Recall 82–88%, ROC-AUC 0.84–0.88

**Optional (Secondary):**
- U-Net segmentation (pixel-level)
- Full-image input (256×256 or 512×512)
- Focal Loss for class imbalance
- Expected: Smoother probability maps; potentially higher lead time

### B. Success Criteria Ranking

1. **Must Have:** Reproducible preprocessing + baseline model + evaluation report
2. **Should Have:** ≥85% recall on test set; error analysis
3. **Nice to Have:** U-Net comparison; live demo; MMD outreach

---

## 14. Sign-off

**Prepared by:** John (Product Manager Agent), BMAD System  
**Reviewed by:** Bryan Chai Wen Cheng  
**Approved by:** [Supervisor Signature/Approval Pending]  

**Next Steps:** 
1. Supervisor review & feedback on PRD (by 2026-05-17)
2. Incorporate revisions
3. Kickoff Phase 1: Data Acquisition (Week 1, starting ~2026-05-20)

---

**END OF PRD**
