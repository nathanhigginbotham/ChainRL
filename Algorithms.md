# Algorithms
A quick, brief summary of the algorithms we are using and how they work. 


## SAC 

Soft Actor Critic (SAC) Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

SAC is the successor of Soft Q-Learning SQL and incorporates the double Q-learning trick from TD3. A key feature of SAC, and a major difference with common RL algorithms, is that it is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy.

## A2C 

A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C). It uses multiple workers to avoid the use of a replay buffer.

## PPO 

The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far from the old policy. For that, ppo uses clipping to avoid too large update.

## ARS

One of the key features of ARS is its ability to handle high-dimensional action spaces, which makes it well-suited for problems with large numbers of possible actions. Additionally, ARS is robust to noisy or sparse reward signals, making it well-suited for challenging reinforcement learning problems.

## RecurrentPPO

## TQC

## TRPO


- A Deep Reinforcement Learning Approach to Supply Chain Inventory Management https://arxiv.org/abs/2204.09603

- Global supply chain management: A reinforcement learning approach https://doi.org/10.1080/00207540110118640

- Reinforcement learning for supply chain optimization http://www.lix.polytechnique.fr/~jread/papers/Kemmer%20et%20al%20-%20Reinforcement%20Learning%20for%20Supply%20Chain%20Optimization.pdf