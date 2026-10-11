"""The controller: GAT + PPO + CTDE.

Layer order: `graph` (topology) -> `features` (state/reward) -> `nets` (GAT encoder,
actor and critic) -> `ppo` (learning) -> `loop` (simulator coupling).
"""
