## Depth and invalid pixels per split / domain
| split | domain | samples | depth p5 | median | p95 | p99 | p99.9 | invalid % (mean) | invalid % (median) | brightness |
|---|---|---|---|---|---|---|---|---|---|---|
| train | indoor | 2606 | 1.0 | 2.6 | 9.0 | 20.1 | 36.6 | 0.7 | 0.2 | 95 |
| train | outdoor | 2389 | 1.4 | 6.5 | 37.9 | 75.4 | 147.8 | 30.3 | 22.7 | 101 |
| val | indoor | 700 | 0.6 | 1.3 | 3.5 | 6.4 | 9.3 | 0.2 | 0.0 | 89 |
| val | outdoor | 400 | 2.1 | 7.7 | 33.1 | 51.6 | 90.9 | 44.4 | 48.3 | 98 |
| test | indoor | 700 | 0.8 | 1.8 | 5.0 | 7.5 | 13.9 | 0.2 | 0.0 | 92 |
| test | outdoor | 400 | 2.5 | 8.5 | 46.1 | 84.0 | 120.8 | 31.8 | 26.4 | 96 |

## Outliers (train)
| domain | threshold | samples with any pixel above | % of all valid pixels above |
|---|---|---|---|
| indoor | > 20 m | 509 / 2606 (19.5%) | 1.0029% |
| indoor | > 30 m | 277 / 2606 (10.6%) | 0.6604% |
| indoor | > 50 m | 0 / 2606 (0.0%) | 0.0000% |
| outdoor | > 100 m | 1106 / 2389 (46.3%) | 0.4270% |
| outdoor | > 150 m | 519 / 2389 (21.7%) | 0.0900% |
| outdoor | > 200 m | 149 / 2389 (6.2%) | 0.0048% |
| outdoor | > 300 m | 0 / 2389 (0.0%) | 0.0000% |

## Vertical angle code per scene (% of samples)
| scene | split | domain | samples | 000 | 010 | 020 | 030 | 040 | 050 | looks up (>= 040) |
|---|---|---|---|---|---|---|---|---|---|---|
| 00002 | train | indoor | 700 | 21 | 19 | 18 | 20 | 12 | 10 | 22% |
| 00003 | train | indoor | 466 | 15 | 14 | 14 | 16 | 19 | 22 | 41% |
| 00004 | train | indoor | 400 | 29 | 22 | 20 | 20 | 7 | 2 | 9% |
| 00005 | train | indoor | 340 | 13 | 12 | 16 | 20 | 18 | 21 | 39% |
| 00006 | train | indoor | 700 | 17 | 16 | 17 | 14 | 16 | 19 | 36% |
| 00007 | train | outdoor | 389 | 17 | 18 | 20 | 20 | 15 | 10 | 25% |
| 00008 | train | outdoor | 400 | 15 | 14 | 15 | 15 | 20 | 22 | 42% |
| 00010 | train | outdoor | 400 | 16 | 18 | 16 | 20 | 15 | 14 | 30% |
| 00013 | train | outdoor | 400 | 15 | 17 | 18 | 16 | 18 | 15 | 33% |
| 00014 | train | outdoor | 400 | 19 | 13 | 20 | 17 | 18 | 13 | 31% |
| 00015 | train | outdoor | 400 | 16 | 16 | 19 | 17 | 17 | 14 | 31% |
| 00001 | val | indoor | 700 | 13 | 13 | 17 | 22 | 18 | 16 | 35% |
| 00017 | val | outdoor | 400 | 18 | 16 | 18 | 16 | 15 | 17 | 32% |
| 00000 | test | indoor | 700 | 16 | 15 | 17 | 17 | 18 | 17 | 35% |
| 00018 | test | outdoor | 400 | 20 | 16 | 16 | 22 | 16 | 9 | 26% |
- train indoor: 29.3% of samples look up
- train outdoor: 31.9% of samples look up
- val indoor: 34.7% of samples look up
- val outdoor: 31.5% of samples look up
- test indoor: 35.4% of samples look up
- test outdoor: 25.8% of samples look up
