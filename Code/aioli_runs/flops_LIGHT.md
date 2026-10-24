| config   | method           |   params |        tokens |             flops | note                                                  |
|:---------|:-----------------|---------:|--------------:|------------------:|:------------------------------------------------------|
| LIGHT    | base (per run)   | 70426624 |   2.94912e+07 | 12461793922252800 | 6*N*D; full per-run budget (incl. Aioli probes)       |
| LIGHT    | DoGE (total)     | 70426624 | nan           | 13209501557587968 | base + 432 alignment micro-batches                    |
| LIGHT    | Skill-It (total) | 70426624 | nan           | 14019518162534400 | base + one-time graph learning: 18 proxies x 50 steps |