# PulseLOB 0.3.0 latency report

CPU: Apple M4, counter 1000.000 MHz (resolution 1.00 ns), timer overhead 7.87 ns (subtracted)
Stream: 600 s of simulated D=4 Hawkes flow, 153628 events (76801 trades, the rest book updates and heartbeats)

## Tick-to-quote service time (ns)

| path                               |    events |    mean |     p50 |     p95 |     p99 |    p99.9 |      max |
|------------------------------------|-----------|---------|---------|---------|---------|---------|----------|
| inline, Cholesky: trades           |     76801 |    1025 |    1056 |    1120 |    1216 |     2880 |    66402 |
| inline, Cholesky: all events       |    153628 |     565 |     912 |    1088 |    1120 |     1504 |    66402 |
| inline, Potter: trades             |     76801 |    3265 |    2752 |    5760 |   13312 |    69632 |   441072 |
| inline, Potter: all events         |    153628 |    1696 |    1888 |    3328 |    7552 |    43008 |   441072 |
| split, hot thread: trades          |     76801 |     295 |     160 |     368 |     416 |     4864 |  3760196 |
| split, hot thread: all events      |    153628 |     253 |     160 |     328 |     416 |     3904 |  3760196 |

## Throughput (closed loop, no timer in the loop)

| path | million events/s |
|---|---|
| inline, Cholesky | 1.78 |
| inline, Potter | 0.67 |
| split, hot thread | 5.77 |
| estimator alone (model steps/s, Cholesky) | 1.14 |

## Component breakdown (mean ns per call, in situ)

Replica state equals the real model byte for byte: Cholesky yes, Potter yes.

| stage | calls | Cholesky | Potter |
|---|---|---|---|
| interval terms: Q+1 exp, psi, F (Thm 3, 7) | 76824 | 127 | 123 |
| compensators + CUSUM accumulation | 76824 | 28 | 25 |
| estimator interval update (Thm 7) | 76824 | 96 | 1885 |
| kernel decay (Thm 2) + long-gap restore | 76824 | 11 | 11 |
| event: regressor, residual, C row | 76801 | 12 | 168 |
| ridge replenish + solve / refresh | 76825 | 546 | 284 |
| serve: Perron, projection, rho~, Jacobi | 76825 | 69 | 65 |
| decision: forecast, toxicity, quotes | 76825 | 94 | 94 |

## Open-loop latency (ns, scheduled arrival to signal, includes queueing)

Poisson arrivals at the given rate; the replay busy-waits for each arrival. Staleness is the number of estimator steps the hot thread's parameters lag the stream when its signal is produced.

| path | rate (events/s) | trades p50 | p99 | p99.9 | max | all p99 | staleness p50 / p99 / max |
|---|---|---|---|---|---|---|---|
| inline, Cholesky | 1e+05 | 1024 | 1888 | 14080 | 116815 | 1440 | n/a |
| inline, Potter | 1e+05 | 2688 | 4480 | 18944 | 131388 | 4096 | n/a |
| split, hot thread | 1e+05 | 192 | 384 | 7808 | 45465 | 376 | 1 / 407 / 535 |
| inline, Cholesky | 1e+06 | 1120 | 4096 | 16896 | 31485 | 3904 | n/a |
| inline, Potter | 1e+06 | 33554432 | 66060288 | 67108864 | 66381288 | 66060288 | n/a |
| split, hot thread | 1e+06 | 232 | 544 | 8704 | 20623 | 528 | 1 / 26 / 73 |

Split mode published 2401 parameter snapshots in the closed-loop run.
