
## Introduction

This repository contains a demo using Python 3 of the theoretical results in the following paper:

[*Decentralized MIMO Systems With Imperfect CSI Using LMMSE Receivers*](https://ieeexplore.ieee.org/abstract/document/10879281)

## How to run

```
python test.py
```

## How to set up

Please use `Config` to set up the system parameters (number of antennas, users, clusters, noise powers, etc.). Please use class `SINRAna` to calculate the SINR. For more details, we kindly refer to test.py. 


Note that the spatial correlation matrices  $`\{\boldsymbol{R}_j\}_{j \leq n}`$ are given. Please rewrite the function `SINRAna.init_correlations` to set up your own correlation matrices.

