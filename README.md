# From SGD to Muon

Code from the "From SGD to Muon Paper", original idea introduced [here](https://massena-t.github.io/blog/2026/02/19/schatten-muon.html).

This is a voluntarily stripped down version of the code for the SMuon optimizer variants, which dynamically recover SGD (or Adam), or Muon using spectral statistics. To make this dynamic interpolation work and reconcile varying geometries with different optimal step sizes we use second-order moments for stable RMS normalized updates, effectively interpolating between Adam and Muon update rules.

Some scripts are provided to reproduce the benchmarking inside the paper (ImageNette, NanoGPT and LoRA experiments). 

Please consider citing our work:

```
TODO
```
