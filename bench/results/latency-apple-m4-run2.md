# PulseLOB 0.3.0 latency report

CPU: Apple M4, counter 1000.000 MHz (resolution 1.00 ns), timer overhead 11.60 ns (subtracted)
Stream: 600 s of simulated D=4 Hawkes flow, 153628 events (76801 trades, the rest book updates and heartbeats)

## Tick-to-quote service time (ns)

| path                               |    events |    mean |     p50 |     p95 |     p99 |    p99.9 |      max |
|------------------------------------|-----------|---------|---------|---------|---------|---------|----------|
| inline, Cholesky: trades           |     76801 |    1006 |     992 |    1088 |    1312 |     2432 |    15673 |
| inline, Cholesky: all events       |    153628 |     554 |     864 |    1088 |    1216 |     2048 |    17130 |
| inline, Potter: trades             |     76801 |    2678 |    2624 |    3264 |    3328 |     5504 |    37463 |
| inline, Potter: all events         |    153628 |    1388 |    1600 |    3264 |    3328 |     4096 |    37463 |
| split, hot thread: trades          |     76801 |     161 |     156 |     200 |     200 |      408 |     2030 |
| split, hot thread: all events      |    153628 |     130 |     114 |     200 |     200 |      368 |     2030 |

## Throughput (closed loop, no timer in the loop)

| path | million events/s |
|---|---|
| inline, Cholesky | 1.73 |
| inline, Potter | 0.72 |
| split, hot thread | 8.39 |
| estimator alone (model steps/s, Cholesky) | 1.17 |

## Component breakdown (mean ns per call, in situ)

Replica state equals the real model byte for byte: Cholesky yes, Potter yes.

| stage | calls | Cholesky | Potter |
|---|---|---|---|
| interval terms: Q+1 exp, psi, F (Thm 3, 7) | 76824 | 117 | 119 |
| compensators + CUSUM accumulation | 76824 | 24 | 21 |
| estimator interval update (Thm 7) | 76824 | 89 | 1882 |
| kernel decay (Thm 2) + long-gap restore | 76824 | 7 | 7 |
| event: regressor, residual, C row | 76801 | 8 | 164 |
| ridge replenish + solve / refresh | 76825 | 510 | 281 |
| serve: Perron, projection, rho~, Jacobi | 76825 | 65 | 61 |
| decision: forecast, toxicity, quotes | 76825 | 90 | 91 |

## Open-loop latency (ns, scheduled arrival to signal, includes queueing)

Poisson arrivals at the given rate; the replay busy-waits for each arrival. Staleness is the number of estimator steps the hot thread's parameters lag the stream when its signal is produced.

| path | rate (events/s) | trades p50 | p99 | p99.9 | max | all p99 | staleness p50 / p99 / max |
|---|---|---|---|---|---|---|---|
| inline, Cholesky | 1e+05 | 1008 | 27136 | 2490368 | 3592241 | 25088 | n/a |
| inline, Potter | 1e+05 | 2688 | 4352 | 10240 | 50109 | 3904 | n/a |
| split, hot thread | 1e+05 | 196 | 65536 | 3604480 | 4980656 | 69632 | 1 / 151 / 538 |
| inline, Cholesky | 1e+06 | 1184 | 671744 | 950272 | 976885 | 671744 | n/a |
| inline, Potter | 1e+06 | 34603008 | 66060288 | 67108864 | 66498984 | 66060288 | n/a |
| split, hot thread | 1e+06 | 224 | 2031616 | 2883584 | 2918533 | 2031616 | 2 / 4607 / 4978 |

Split mode published 2401 parameter snapshots in the closed-loop run.
