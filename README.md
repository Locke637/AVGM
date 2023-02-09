# AVGM: Adaptive Value Decomposition with Greedy Marginal Contribution Computation for Cooperative Multi-Agent Reinforcement Learning

## Overview

This project is based on [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms).  Adaptive Value decomposition with Greedy Marginal contribution (AVGM) is based on an adaptive value decomposition that learns the cooperative value of a group of dynamically changing agents. We first illustrate that the proposed value decomposition can consider the complicated interactions among agents and is feasible to learn in large-scale scenarios. Then, our method uses a greedy marginal contribution computed from the value decomposition as an individual credit to incentivize agents to learn the optimal cooperative policy. We further extend the module with an action encoder to guarantee the linear time complexity for computing the greedy marginal contribution.

## Source Code

### Requirement

- python
- torch
- [MAgent](https://github.com/geek-ai/MAgent)

### Quick Start

```shell
$ python main_magent.py
```

## Credits

For researchers that have leveraged or compared to this work, please cite the following:

Adaptive Value Decomposition with Greedy Marginal Contribution Computation for Cooperative Multi-Agent Reinforcement Learning. Shanqi Liu, Yujing Hu, Runze Wu, Dong Xing, Yu Xiong, Changjie Fan, Kun Kuang and Yong Liu, AAMAS2023 (Full Paper).

## License

The code is provided under the [MIT License](https://opensource.org/licenses/MIT).
