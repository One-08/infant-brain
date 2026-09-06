# Results

Place reproducible benchmark summaries, evaluation metrics, and figures here.

Keep large model checkpoints, virtual environments, caches, and temporary logs
out of this directory. Record the command, configuration, seed, device, and
software version alongside each result so others can reproduce it.

## Breakout benchmark

The Breakout benchmark compares the full system with a baseline and three
ablations across 200 CUDA episodes. The plotted lines show smoothed episode
reward over time. This is a quick, single-run comparison: it does not report
mean +/- standard deviation across seeds or a significance test. The raw
traces and chart are here:

![Breakout benchmark](benchmark/proof_benchmark.png)

- Chart: [proof_benchmark.png](benchmark/proof_benchmark.png)
- Raw combined data: [all_results.json](benchmark/all_results.json)
- Individual runs: [baseline.json](benchmark/baseline.json),
  [full_brain.json](benchmark/full_brain.json),
  [full_brain_(s2).json](benchmark/full_brain_(s2).json),
  [no_intuition.json](benchmark/no_intuition.json),
  [no_slots.json](benchmark/no_slots.json), and
  [no_stages.json](benchmark/no_stages.json)

In this comparison, **No Intuition** disables the self-monitoring
`IntuitionGate`; it does not remove the separate curiosity exploration drive.