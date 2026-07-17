# Final Defence Q&A ? Version 2

## Why was Version 1 flawed?
It was computationally reproducible but allowed shortcuts from geography, frame selection and negative sampling.

## Why retain Version 1?
It is audit evidence showing how the methodology evolved and why Version 2 was necessary.

## Why did the small CNN beat ResNet-50?
The compact CNN is trained directly on multispectral meteorological patches, while frozen ImageNet features are not necessarily aligned with infrared cloud texture.

## Why is natural-prevalence accuracy lower?
It reflects a less artificial class mix and more realistic zero-recorded frames, not balanced controlled sampling.

## Why did recall fall to about 69%?
The frozen validation-selected threshold is conservative; it keeps false alarms low but misses 185 of 602 recorded-positive natural-prevalence patches.

## Does zero-recorded mean no lightning?
No. It means no MMD-recorded cloud-to-ground strike under the frozen rule.

## Is this operational?
No. It is a research classifier for recorded associations, not a real-time warning system.

## Why does the geographic baseline still reach 0.817 ROC-AUC?
Lightning occurrence and MMD recording are geographically and temporally structured; residual predictability remains.

## Why PR-AUC?
It is more informative under imbalanced or natural-prevalence settings where positive precision is central.

## Why date/storm-disjoint splits?
They reduce leakage from near-duplicate frames and storm persistence across train and test.

## How was test leakage prevented?
Thresholds, calibration, loss/augmentation selection and model choice were frozen from validation before the unlock commit; test inference occurred afterward.

## Does it generalize outside Peninsular Malaysia?
Not established. The claim is limited to the empirical study region.

## What MMD metadata would improve the study?
Official coverage/completeness masks, sensor-status metadata, independent event IDs, quality flags and richer storm annotations.
