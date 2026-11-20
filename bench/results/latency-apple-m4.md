# PulseLOB 0.3.0 latency report

CPU: Apple M4, counter 1000.000 MHz (resolution 1.00 ns), timer overhead 14.49 ns (subtracted)
Stream: 600 s of simulated D=4 Hawkes flow, 153628 events (76801 trades, the rest book updates and heartbeats)

## Tick-to-quote service time (ns)

| path                               |    events |    mean |     p50 |     p95 |     p99 |    p99.9 |      max |
|------------------------------------|-----------|---------|---------|---------|---------|---------|----------|
| inline, Cholesky: trades           |     76801 |     994 |     992 |    1088 |    1120 |     2112 |    10045 |
| inline, Cholesky: all events       |    153628 |     547 |     912 |    1088 |    1120 |     1504 |    17069 |
| inline, Potter: trades             |     76801 |    2694 |    2624 |    3264 |    3328 |     7552 |   100163 |
| inline, Potter: all events         |    153628 |    1395 |    1600 |    3264 |    3328 |     7040 |   100163 |
| split, hot thread: trades          |     76801 |     158 |     156 |     196 |     196 |      368 |    10170 |
| split, hot thread: all events      |    153628 |     127 |     112 |     196 |     196 |      368 |    10170 |

## Throughput (closed loop, no timer in the loop)

| path | million events/s |
|---|---|
| inline, Cholesky | 1.87 |
| inline, Potter | 0.73 |
| split, hot thread | 8.45 |
| estimator alone (model steps/s, Cholesky) | 1.15 |

## Component breakdown (mean ns per call, in situ)

Replica state equals the real model byte for byte: Cholesky yes, Potter yes.

| stage | calls | Cholesky | Potter |
|---|---|---|---|
| interval terms: Q+1 exp, psi, F (Thm 3, 7) | 76824 | 116 | 121 |
| compensators + CUSUM accumulation | 76824 | 21 | 18 |
| estimator interval update (Thm 7) | 76824 | 87 | 1867 |
| kernel decay (Thm 2) + long-gap restore | 76824 | 4 | 4 |
| event: regressor, residual, C row | 76801 | 6 | 160 |
| ridge replenish + solve / refresh | 76825 | 538 | 275 |
| serve: Perron, projection, rho~, Jacobi | 76825 | 62 | 58 |
| decision: forecast, toxicity, quotes | 76825 | 87 | 87 |

## Open-loop latency (ns, scheduled arrival to signal, includes queueing)

Poisson arrivals at the given rate; the replay busy-waits for each arrival. Staleness is the number of estimator steps the hot thread's parameters lag the stream when its signal is produced.

| path | rate (events/s) | trades p50 | p99 | p99.9 | max | all p99 | staleness p50 / p99 / max |
|---|---|---|---|---|---|---|---|
| inline, Cholesky | 1e+05 | 1024 | 1312 | 2752 | 41265 | 1216 | n/a |
| inline, Potter | 1e+05 | 2688 | 4352 | 7424 | 50909 | 3776 | n/a |
| split, hot thread | 1e+05 | 192 | 440 | 11264 | 95180 | 416 | 1 / 55 / 493 |
| inline, Cholesky | 1e+06 | 1152 | 303104 | 401408 | 449069 | 303104 | n/a |
| inline, Potter | 1e+06 | 33030144 | 65011712 | 65011712 | 65010390 | 65011712 | n/a |
| split, hot thread | 1e+06 | 232 | 752 | 28672 | 82862 | 736 | 1 / 77 / 185 |

Split mode published 2401 parameter snapshots in the closed-loop run.
