# Project Context - Lightning Flash Classification

This file is auto-loaded by BMAD agents and workflows through persistent_facts.

## Primary Reference

Primary source of truth for this project is AGENT_REFERENCE_ARTIFACT.md at project root.
When planning, designing, coding, reviewing, or testing, use that artifact as the main context document.
If details here and the artifact differ, trust AGENT_REFERENCE_ARTIFACT.md.

## Always-In-Scope Project Facts

- Project name: Lightning Flash Classification
- Goal: Predict cloud-to-ground lightning for Malaysia in a 0-60 minute nowcasting window
- Inputs: Himawari-8 satellite imagery (IR, WV, VIS) plus MMD lightning ground truth
- Core model: ResNet-50 with focal loss for class imbalance
- Key success metric: recall target >= 85 percent on test data

## Current Phase

- Current phase: Phase 4 - testing and daily ingestion infrastructure
- Core ML pipeline and tests are complete
- Real-data integration is in progress and depends on MMD CSV plus Himawari archive

## Implementation Constraints

- Prefer existing project structure and naming in src/, tests/, data/, models/, results/, logs/
- Keep scheduler and logging output Windows-safe (ASCII-friendly)
- Preserve duplicate-prevention behavior in daily ingestion metadata flow
- Keep albumentations compatibility pinned to 1.3.1 behavior expectations

## Working Rule For BMAD Agents

Before giving recommendations or edits, align with AGENT_REFERENCE_ARTIFACT.md and then apply role-specific BMAD behavior.
