| config   | method           |    params |       tokens |              flops | note                                                   |
|:---------|:-----------------|----------:|-------------:|-------------------:|:-------------------------------------------------------|
| HEAVY    | base (per run)   | 162322944 |   2.3593e+08 | 229780723492454400 | 6*N*D; full per-run budget (incl. Aioli probes)        |
| HEAVY    | DoGE (total)     | 162322944 | nan          | 236674145197228032 | base + 432 alignment micro-batches                     |
| HEAVY    | Skill-It (total) | 162322944 | nan          | 287225904365568000 | base + one-time graph learning: 18 proxies x 200 steps |