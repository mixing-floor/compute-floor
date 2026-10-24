| config             | method           |   params |       tokens |              flops | note                                                      |
|:-------------------|:-----------------|---------:|-------------:|-------------------:|:----------------------------------------------------------|
| pythia70m_heavy_s2 | base (per run)   | 70426624 |   2.3593e+08 |  99694351378022400 | 6*N*D; full per-run budget (incl. Aioli probes)           |
| pythia70m_heavy_s2 | DoGE (total)     | 70426624 | nan          | 103183653676253184 | base + 504 logged alignment micro-batches across 7 rounds |
| pythia70m_heavy_s2 | Skill-It (total) | 70426624 | nan          | 124617939222528000 | base + one-time graph learning: 18 proxies x 200 steps    |