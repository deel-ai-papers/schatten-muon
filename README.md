# From SGD to Muon

Code from the ["From SGD to Muon"](https://hal.science/hal-05626222) paper, now accepted to NeurIPS 2026.

If you prefer shorter reads, I initially wrote a [blogpost](https://massena-t.github.io/blog/2026/02/19/schatten-muon.html) that lays out the main concepts behind the subject, although that was before I started optimizing the originally naïve method.

This is a voluntarily stripped down version of the code for the SMuon optimizer variants, which dynamically recover SGD (or Adam), or Muon using spectral statistics. To make this dynamic interpolation work and reconcile varying geometries with different optimal step sizes we use second-order moments for stable RMS normalized updates, effectively interpolating between Adam and Muon update rules.

Some scripts are provided to reproduce the benchmarking inside the paper (ImageNette, NanoGPT and LoRA experiments). 

Please consider citing our work:

```
@inproceedings{massena2026sgdmuon,
  title     = {From SGD to Muon: Adaptive Optimization via Schatten-p Norms},
  author    = {Massena, Thomas and Friedrich, Corentin and Serrurier, Mathieu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://hal.science/hal-05626222}
}
```
