# V2 Test-Lock Chronology Audit

Verdict: **CONFIRMED**.

| Commit | Role | Git record |
| --- | --- | --- |
| `1758f98` | validation selection | `1758f9874954479d79ec9dcb271f5faa0a776089 2026-07-16T19:35:03+08:00 Record Version 2 Phase 3 validation selection` |
| `94dc243` | test unlock | `94dc2438bb69f89069f14d8c46d360167425789d 2026-07-16T19:35:55+08:00 Record Version 2 Phase 3 test unlock` |
| `b8062b8` | final evaluation reports | `b8062b824978b60fa34880cd0a2dc6d75b250f41 2026-07-16T19:53:50+08:00 Record Version 2 Phase 3 final evaluation` |
| `72d5b86` | artifact generator | `72d5b86895a280850f69bda2702d103a06da4638 2026-07-16T19:56:26+08:00 Add Version 2 Phase 3 artifact generator` |

Execution bug-fix commits `21dfee8`, `c4affc3`, `e5ae784`, `400ce32`, and `6744c57` changed execution/resume/checkpoint/reporting mechanics only; they did not change seeds, thresholds, calibration, model definitions, labels, splits or scientific selection rules.
