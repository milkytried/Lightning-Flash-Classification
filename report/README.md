# Report package index

The final FYP report supplied with the submission is the narrative source of truth for the aligned Himawari-9 / MMD experiment. The repository-facing summary, exact commands, and final performance values are in [../README.md](../README.md).

The final numerical record is [../results/satellite_frozen_cpu_clean_metrics.json](../results/satellite_frozen_cpu_clean_metrics.json). The checkpoint `models/satellite_resnet50_frozen_cpu_clean_best.pth`, dataset manifest `data/processed/satellite_dataset.csv`, derived patches, and generated Figure 5.1–5.8 PNG files under `results/figures/` are gitignored generated artifacts; recreate them with the sequence in the root README.

Superseded Himawari-8 prototype reports, status notes, the old audit, and the old viva material are preserved in [../docs/archive/](../docs/archive/README.md). They document the 11-PNG baseline only and must not be used as the final FYP result.
