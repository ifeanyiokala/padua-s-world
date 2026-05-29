# Specification: MiniGrid-Causal v0

## 1. Overview
**Environment Name:** `MiniGrid-Causal-v0`
**Base Architecture:** `MiniGrid-Empty-8x8`
**Objective:** To evaluate the causal robustness of world models (e.g., DreamerV3) against spurious visual correlations in a fully observable, discrete spatial environment.

## 2. Environment Dynamics
The environment consists of an 8x8 enclosed room. The agent spawns in the absolute center of the grid facing a random orientation. 

### The Causal Target (The Goal)
* **Mechanism:** A single Goal tile (e.g., a green square) is spawned at the midpoint of one of the four bounding walls (North, South, East, or West).
* **True Causal Rule:** The optimal sequence of actions is strictly determined by the spatial coordinate of this Goal tile.

### The Confounding Variable (Spurious Feature)
* **Mechanism:** The background floor tiles of the entire 8x8 grid are painted a uniform color selected from a predefined set: `[Red, Blue, Yellow, Purple]`.
* **The Trap:** The floor color holds no causal power over the environment's physics, reward function, or the agent's spatial coordinates. It is a purely visual artifact.

## 3. Distribution Shift Protocol
To test for causal confusion, the environment enforces a strict statistical dependency between the Goal Direction and the Floor Color during training, which is subsequently severed during testing.

### Training Regime (Correlated)
The floor color is spuriously correlated with the goal direction 80% of the time. The remaining 20% introduces random noise to prevent immediate mode collapse.
* Goal is North -> 80% chance floor is Red.
* Goal is South -> 80% chance floor is Blue.
* Goal is East -> 80% chance floor is Yellow.
* Goal is West -> 80% chance floor is Purple.

### Testing Regime (Uncorrelated / Randomized)
The spurious correlation is completely removed. Floor color is sampled uniformly at random regardless of the goal location.
* P(Color | Goal Direction) = 0.25 for all combinations.

## 4. Observation and Action Spaces
* **Observation Space:** Fully observable pixel rendering (64x64x3 RGB array) tailored for DreamerV3's visual encoder.
* **Action Space:** Discrete (Left, Right, Forward).
* **Reward Function:** * +1.0 for reaching the Goal tile.
  * -0.01 step penalty to encourage shortest-path navigation.
  * 0.0 for episode termination via timeout.

## 5. Success Criteria
A successful causally-robust model will achieve near-identical episodic returns in both the Training and Testing regimes. If the model incorrectly attends to the spurious floor color to predict value or dynamics, its performance will catastrophically degrade during the Testing regime when P(Color | Goal) shifts.
