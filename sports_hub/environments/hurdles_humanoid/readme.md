<div align="center">
  <h1>Hurdles Humanoid</h1>
</div>

<img width="1818" height="730" alt="humanoid_hurdles" src="https://github.com/user-attachments/assets/73df4c68-7274-4cb2-8104-c62c870cb11a" />

<br/>

<p align="center">
  <a href="https://i1cps.github.io/Sports-Hub-Website/"><strong>🔗 Click for live demo </strong></a>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧩 Difficulty

<p align="center" style="font-size:1.1em">
⭐⭐⭐⭐⭐⭐⭐⭐☆ ☆<br/>
<em>8/10: Purely uses LiDAR rays for scene perception </em>
</p>

A sprint across a series of hurdles 0.6m high using a set of LiDAR rays for spatial information. Momentum, mid-jump body posture and timing are key

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧮 Reward Function

Each timestep the agent receives a reward composed of three terms:

- 🏃‍♂️ **Forward Velocity**  
  Encourages the humanoid to sprint forward along the x‑axis.

- ❤️ **Healthy Bonus**  
  If the humanoid's torso is above 1.0m it receives a healthy reward, otherwise it receives an unhealthy penalty.

- ⚙️ **Control Cost**  
  Penalizes large actuator torques to promote efficient, smooth motions.

- 🤕 **Collision Penalty**  
  Penalize the humanoid every time it hits a hurdle.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🔁 Reset Logic

Each of the **4096 parallel environments** are initialized with small uniform noise [-0.01, +0.01] applied to joint positions and velocities of the humanoid. This encourages robustness and prevents agents from overfitting to a single deterministic start state.

The episode resets early if the agent **falls below a height threshold**, i.e. it tips over or collapses. This drastically speeds up learning.

```python
  def _compute_termination(self, data: mjx.Data) -> jax.Array:
      # Get the height of the humanoids torso, make sure its above the height limit
      torso_height = data.site_xpos[self._agent_site_index][2]
      upright = torso_height > self._agent_healthy_height

      reset_condition = jp.logical_not(upright)
      return jp.where(reset_condition, 1.0, 0.0)
```

> In evaluation mode, the environment also resets when the agent **crosses the finish line**, simulating a race format.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 👁 Observation Function

The agent receives a **279‑dimensional** observation vector:

- **22 Joint Positions**  
  Local joint angles, root height, and orientation (quat)

- **23  Joint Velocities**  
  Angular and linear velocity terms for each joint.

- **140  Body Inertias**  
  Flattened inertial properties (`cinert`) of each body element.

- **84  Body Velocities**  
  World‑frame linear and angular velocities (`cvel`) for each body.

- **23  Actuator Forces**  
  The force applied to each DoF by the combined effect of all actuators (`qfrc_actuator`).

- **84 External Forces**
  Flattened external forces applied to each body of the humanoid clipped to the range -1.0 and 1.0 (`cfrc_ext`).

- **30 LiDAR Rays**
  Distance measurements to nearby obstacles in fixed directions.

> 🗒️ **Total:** 22 + 23 + 140 + 84 + 23 + 84 + 30 = **406** dimensions

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🏗️ MJCF Highlights

Some notable design choices in the `model.xml`:

- 👁️ LiDAR-Based Perception (30 rays)  
  The humanoid’s head is modelled with an array of 30 rangefinder rays. These simulate a LiDAR-style depth scan.

> 💡 The outputs from the LiDAR rays are processed using a hyperbolic tangent (`tanh`) activation function.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📊 Hyperparameters

The following settings (from `hyperparams.json`) were used to train the Hurdles Humanoid with PPO:

| Parameter               | Value           |
| ----------------------- | --------------- |
| Total timesteps         | 360,000,000     |
| Learning rate           | 0.5 × 10⁻⁴      |
| Num environments        | 4,096           |
| Unroll length           | 10 steps        |
| Batch size              | 512             |
| Episode length          | 1,000 steps     |
| Num minibatches         | 32              |
| Updates per batch       | 8               |
| Discount factor (γ)     | 0.99            |
| Entropy cost            | 0.01            |
| Max gradient norm       | 1.0             |
| PPO clipping ε          | 0.2             |
| Reward scaling          | 0.1             |
| Action repeat           | 1               |
| Normalize observations  | True            |
| RNG seed                | 14              |
| Num evaluation rollouts | 10              |
| Return best params      | True            |
| **Policy network**      | [512, 256, 128] |

> 🧠 The policy network uses a deeper architecture: 3 layers with 512, 256, and 128 hidden units. This helps the agent understand this more complex environment.

These are tuned for fast, stable PPO convergence

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

Ready to try Hurdles? This environment is a solid demonstration of the power of LiDAR rays in locomotion
