# Version 2 Full Dataset Validation

All results use frozen configuration `eebe246103c5543f1c2b4618d76f91fe678edf14d966b19842c1e02b5928b081`, ledger `3d7545d868cace4aea0750d0f235a32e3375e3fa00cfa6eaf9f066b761781a1a`, and manifest `a0beb139f3b654028938952af39c29755fef7259547b01ba9dd58898fbfb585a`.

## Controlled dataset

| index | patches | positive | negative | dates | frames | positive_frames | storms | active_dates | positive_per_frame_median | positive_per_frame_max | largest_frame_fraction | largest_date_fraction | largest_storm_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2745.0 | 758.0 | 1987.0 | 32.0 | 229.0 | 146.0 | 191.0 | 30.0 | 5.0 | 18.0 | 0.008014571948998178 | 0.24408014571949 | 0.06156648451730419 |
| train | 9204.0 | 3169.0 | 6035.0 | 223.0 | 708.0 | 458.0 | 628.0 | 167.0 | 7.0 | 18.0 | 0.0023902651021295088 | 0.030638852672750978 | 0.015319426336375489 |
| val | 2612.0 | 629.0 | 1983.0 | 59.0 | 226.0 | 143.0 | 213.0 | 44.0 | 4.0 | 14.0 | 0.007656967840735069 | 0.03828483920367534 | 0.027182235834609495 |

## Temporal contamination

| index | contaminated negatives |
| --- | --- |
| same_frame | 0 |
| minus10_plus20 | 0 |
| primary_minus20_plus30 | 0 |
| minus30_plus40 | 926 |

## Cross-split overlap

| index | count |
| --- | --- |
| date | 0 |
| frame_id | 0 |
| storm_id | 0 |
| path | 0 |
| sha256 | 0 |
| source_file | 0 |
| crop | 0 |
| within_split_duplicate_crop | 0 |

## Patch quality

| index | count |
| --- | --- |
| missing | 0 |
| corrupt | 0 |
| wrong_dimensions | 0 |
| wrong_channels | 0 |
| constant | 0 |
| black | 0 |
| nonfinite | 0 |
| hash_mismatch | 0 |
| duplicate_file_hash | 0 |
| duplicate_crop | 0 |
| manifest_file_mismatch | 0 |
| incorrect_split_path | 0 |
| incorrect_label_metadata | 0 |

## Distribution matching

| index | SMD | JS | PSI |
| --- | --- | --- | --- |
| centre_lat | 0.09188818089445194 | 0.01247708263722885 | 0.10131809276362191 |
| centre_lon | 0.06764683268016511 | 0.014135818618800283 | 0.1141006595021106 |
| local_hour | 0.18912587444662454 | 0.089035678488975 | 0.7960976277229167 |
| month | 0.22842231956386805 | 0.01558168019558368 | 0.12767726556980188 |
| distance_to_study_mask_boundary_km | 0.33021415475597216 | 0.022105376960026544 | 0.17908857242139278 |
| geographic_grid_2d | nan | 0.033393464875141735 | nan |

Local-hour absolute SMD is 0.189, below the 0.20 target. Geographic-grid JS divergence is 0.0334 versus 0.0367 in Phase 1B. No verified land/ocean mask was available.

## Natural-prevalence evaluation set

The fixed unbalanced grid contains 2,475 eligible patches from 235 frames: 602 positive and 1,873 negative (24.32% recorded-positive prevalence). 343 ambiguous cells were excluded under the preregistered label rule. Full category, month, and time-period breakdowns are in the JSON report.

## Build cost and state

The controlled build took 19,810 seconds (5.50 hours), peaked at 541 MB RSS, and produced 73.6 MB of controlled patches. Prefetch plus direct-builder ledgers record 13.86 GB of source downloads. State-safe completed, failed, and download ledgers are preserved.

## Remaining limitations

- Controlled patch count is 14,561, below the suggested 25,000-40,000 range; independent-frame diversity and per-frame caps were preserved instead of inflating correlated crops.
- Eleven controlled ledger frames and four natural-grid frames were unavailable; all are recorded.
- No reliable coastline/land-ocean mask was available, so none was introduced.
- Storm groups are derived DBSCAN analytical clusters, not official meteorological storm identifiers.
- Overall distance-to-mask-boundary SMD is 0.330 and month SMD is 0.228; geographic-grid JS and latitude/longitude SMDs remain low.
- The largest test-date contribution is 24.4%, below but close to the preregistered 25% dominance limit.
- Natural prevalence is prevalence of MMD-recorded positives under the frozen grid/rule, not true physical lightning prevalence.
