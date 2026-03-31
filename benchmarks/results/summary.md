# Qwen 0.6B benchmark comparison

| Benchmark | Primary metric | Base | Base pass/total | Finetuned | Finetuned pass/total | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IFEval | prompt_level_loose_acc,none | 0.316081 | 171/541 | 0.142329 | 77/541 | -0.173752 |
| IFBench | prompt_level_loose_accuracy | 0.110000 | 33/300 | 0.276667 | 83/300 | +0.166667 |
| BigBenchHard | exact_match,get-answer | 0.001536 | 10/6511 | 0.000000 | 0/6511 | -0.001536 |
| AGI Eval English | acc,none | 0.189772 | 731/3852 | 0.167965 | 647/3852 | -0.021807 |
