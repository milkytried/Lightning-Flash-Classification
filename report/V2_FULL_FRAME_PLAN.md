# Version 2 Phase 2 full frame plan

The temporal label rule was frozen before sampling: positive `[t,t+10m)`, negative exclusion `[t-20m,t+30m)`, full 64x64 crop plus 10 km margin. The frame list is frozen before cache/download decisions.

Configuration hash: `eebe246103c5543f1c2b4618d76f91fe678edf14d966b19842c1e02b5928b081`. Frozen ledger SHA-256: `3d7545d868cace4aea0750d0f235a32e3375e3fa00cfa6eaf9f066b761781a1a`. Frames: 1200; categories {'active': 780, 'zero_recorded': 420}.

## Split support

- test: {'frames': 240, 'active_frames': 156, 'zero_recorded_frames': 84, 'active_dates': 30, 'storm_groups': 118, 'planned_positive_capacity': 1259, 'cached_frames': 21, 'download_frames': 219}
- train: {'frames': 720, 'active_frames': 468, 'zero_recorded_frames': 252, 'active_dates': 170, 'storm_groups': 387, 'planned_positive_capacity': 5100, 'cached_frames': 135, 'download_frames': 585}
- val: {'frames': 240, 'active_frames': 156, 'zero_recorded_frames': 84, 'active_dates': 45, 'storm_groups': 141, 'planned_positive_capacity': 1070, 'cached_frames': 40, 'download_frames': 200}

Cache was checked only after science-first selection. 1004 desired frames require download. Blockers: none. Derived storm groups are not official identifiers.
