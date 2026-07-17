# Version 2 Dataset Card

**Permanent label:** Version 2 ? Frozen Corrected Scientific Experiment

## Sources
- Lightning: Malaysian Meteorological Department cloud-to-ground records, licensed/private and not redistributed.
- Satellite: Himawari-9 AHI Level-1b data accessed from the public NOAA archive; derived patches are generated locally and not committed.

## Study Region and Period
A conservative empirical Peninsular Malaysia rectangle: latitude 1.2?6.8, longitude 99.7?104.4. The full ledger spans the frozen 2023?2025 Version 2 design.

## Labels
- Positive: MMD-recorded cloud-to-ground strike association within the configured 10-minute Himawari frame.
- Negative: no MMD-recorded ground strike in the full crop neighborhood under the frozen exclusion window.
- Temporal exclusion: `[t?20m,t+30m)`, start inclusive, end exclusive.
- Safety margin: 10 km.
- Wording: zero-recorded means no MMD-recorded strike under this rule, not physical absence of lightning.

## Splits and Grouping
Date/storm-disjoint chronological split with derived DBSCAN storm groups. No date, frame, storm, source-file, path, image-hash or crop overlap was detected across splits.

## Composition
| Split | Patches | Positive | Negative | Dates | Frames | Storms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 9204 | 3169 | 6035 | 223 | 708 | 628 |
| Validation | 2612 | 629 | 1983 | 59 | 226 | 213 |
| Controlled test | 2745 | 758 | 1987 | 32 | 229 | 191 |

Natural-prevalence set: 2475 patches, 602 positive, 1873 negative, prevalence 0.243232; 343 ambiguous grid cells excluded.

## Sampling and Matching
Version 2 includes active and zero-recorded frames, full-crop negative exclusion, no-data rejection, deterministic frame selection before cache checks, and frozen manifests.

## Missing Coverage and Restrictions
Eleven controlled frames and four natural-grid frames were unavailable and recorded. MMD records and generated patches/checkpoints may be subject to licensing and size restrictions; raw and generated data are gitignored.

## Known Biases
Residual geography/time predictability remains; the study mask is empirical; MMD detection completeness is not uniform; natural prevalence is prevalence of recorded positives under the frozen rule, not true physical lightning prevalence.
