---
title: "Deep Learning for Lightning Flash Classification Using Geostationary Satellite Imagery"
source: "docs/proposal-extracted.txt"
status: "draft-from-proposal"
created: "2026-05-10"
---

# Product Brief

## Project Summary

This capstone project will develop a deep learning approach for short-term cloud-to-ground lightning nowcasting over Malaysia using geostationary satellite imagery. The system will use Himawari-8 Advanced Himawari Imager data as the primary input and historical ground lightning strike records, preferably from the Malaysian Meteorological Department, as ground truth labels.

The intended output is a model that estimates the probability of cloud-to-ground lightning occurrence within a short lead-time window, up to 0-60 minutes, either for image patches or spatial pixels. The project is positioned as a research and prototype implementation rather than a production public-warning system.

## Problem

Malaysia has high lightning risk, but timely lightning nowcasting remains difficult. Numerical weather prediction struggles with the fine spatiotemporal dynamics of thunderstorm electrification, ground lightning detection networks are mainly observational rather than predictive, and geostationary lightning imagers such as GOES-R GLM are not available over the Asian domain.

The project asks whether existing geostationary satellite imagery can reveal precursor cloud features that allow a deep learning model to nowcast cloud-to-ground lightning over Malaysia with useful lead time.

## Target Users and Stakeholders

- Primary academic stakeholder: capstone supervisor and assessment panel.
- Domain stakeholders: meteorological analysts, MMD or similar local authorities.
- Potential future beneficiaries: event organizers, road workers, aviation operators, infrastructure operators, and the Malaysian public.

## Objectives

1. Acquire and integrate historical lightning data with corresponding Himawari-8 satellite imagery for Malaysia and neighboring regions.
2. Preprocess satellite imagery and lightning records into a supervised learning dataset, including spatial-temporal alignment, labeling, and handling of class imbalance.
3. Develop CNN-based models for lightning nowcasting, beginning with practical baselines such as ResNet-50 or VGG-style classifiers and potentially extending to U-Net-style probability maps.
4. Evaluate model performance using general ML metrics and meteorology-specific verification metrics.
5. Document deliverables, deployment considerations, limitations, and user guidance.

## Scope

In scope:

- Cloud-to-ground lightning nowcasting over Malaysia.
- Himawari-8 AHI satellite imagery, likely using infrared, water vapor, and possibly visible channels.
- Ground lightning records with timestamp and location.
- Patch-based classification or pixel-level segmentation/probability mapping.
- Lead-time window up to 0-60 minutes.
- Python implementation using TensorFlow/Keras as the preferred framework, with PyTorch as an optional comparison path.
- Evaluation using train, validation, and test splits separated by time where possible.

Out of scope:

- Long-range lightning forecasting.
- Intra-cloud lightning type classification.
- Multi-source modeling using radar, weather stations, or numerical model outputs.
- Fully operational real-time deployment.
- Public safety decision automation without expert meteorological review.

## Data Plan

The project expects two main data sources:

- Lightning data: MMD Lightning Detection System records, ideally covering 2-3 years such as 2018-2020, with at least time and latitude/longitude for cloud-to-ground strikes.
- Satellite data: Himawari-8 AHI imagery from JMA or NOAA archives, cropped to Malaysia and nearby regions. Candidate bands include 10.4 micrometer infrared cloud-top temperature, 6.2 micrometer water vapor, and 0.64 micrometer visible imagery for daytime scenes.

Labels will associate a satellite frame at time `t` with lightning events occurring in a future window such as `t` to `t+60 minutes`. Candidate labeling strategies include per-pixel labels or patch labels, for example 64x64 pixel patches centered on or near lightning events.

## Model Approach

The practical baseline should be a patch-based CNN classifier because it is easier to implement, train, and evaluate within a capstone timeline. ResNet-50 transfer learning is a strong first model, with a smaller custom CNN as a sanity-check baseline.

If time and data density allow, the project can extend to a U-Net or fully convolutional architecture that outputs a probability map for lightning risk across the spatial domain.

Class imbalance should be handled through a combination of negative sampling, class weights, focal loss, threshold tuning, and careful metric selection.

## Evaluation

Core metrics:

- Accuracy
- Precision
- Recall / Probability of Detection
- F1-score
- ROC-AUC

Meteorological metrics:

- Probability of Detection
- False Alarm Ratio
- Heidke Skill Score
- True Skill Statistic

Evaluation should emphasize missed lightning events because false negatives carry safety risk, while still monitoring false alarms so the model remains useful.

## Risks and Constraints

- Delayed or restricted access to MMD lightning data.
- Large satellite data volume and preprocessing complexity.
- Class imbalance between lightning and non-lightning samples.
- Geolocation alignment issues, including parallax from high cloud tops.
- Limited GPU resources and training time.
- Risk that the model learns dataset artifacts rather than meteorologically meaningful features.
- Ethical risk if prototype outputs are interpreted as operational warnings.

## Planned Deliverables

- Cleaned and documented dataset preparation pipeline.
- Training-ready labeled dataset or reproducible dataset build scripts.
- Baseline CNN model and training pipeline.
- Evaluation scripts and metric reports.
- Visual results such as probability maps, charts, false positive/false negative examples, and overlays.
- Final report and presentation material.
- User/developer documentation describing how to reproduce experiments and interpret limitations.

## BMAD Recommendation

The next workflow should be `bmad-create-prd` to convert this proposal and brief into implementation requirements. Because this is a research-heavy capstone, the PRD should treat "product" as a research software system with reproducible data processing, experiment tracking, model training, evaluation, and reporting deliverables.
