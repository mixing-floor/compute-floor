| config      | method           |   params |   tokens |           flops | note                                                 |
|:------------|:-----------------|---------:|---------:|----------------:|:-----------------------------------------------------|
| SUPER_LIGHT | base (per run)   | 70426624 |   589824 | 249235878445056 | 6*N*D; full per-run budget (incl. Aioli probes)      |
| SUPER_LIGHT | DoGE (total)     | 70426624 |      nan | 623089696112640 | base + 216 alignment micro-batches                   |
| SUPER_LIGHT | Skill-It (total) | 70426624 |      nan | 405008302473216 | base + one-time graph learning: 18 proxies x 5 steps |