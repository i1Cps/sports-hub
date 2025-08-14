<div align="center">
  <h1>Sprint Humanoid</h1>
</div>

<img width="1818" height="730" alt="Untitled design" src="https://github.com/user-attachments/assets/8dd6465b-d1b0-40fc-ba24-172f517f63f6" />

<br/>

<p align="center">
  <a href="https://i1cps.github.io/Sports-Hub-Website/"><strong>🔗 Click for live demo </strong></a>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧩 Difficulty

<p align="center" style="font-size:1.1em">
⭐☆☆☆☆☆☆☆☆☆<br/>
<em>1/10: Simple joint coordination in a straight line</em>
</p>

This is one of the simplest environments. The humanoid just needs to learn how to move its joints rhythmically to build momentum and sprint in a straight line, no turning, balancing objects, or reacting to obstacles.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


## 🧮 Reward Function

Each timestep the agent receives a reward composed of three terms:

- 🏃‍♂️ **Forward Velocity**  
  Encourages the humanoid to sprint forward along the x‑axis.

- ❤️ **Alive Bonus**  
  A constant bonus awarded per timestep.
  
- ⚙️ **Control Cost**  
  Penalizes large actuator torques to promote efficient, smooth motions.

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

The agent receives a **276‑dimensional** observation vector containing only locomotion-relevant information:

- **22  Joint Positions**   
  Local joint angles, root height and orientation (quat)

- **23  Joint Velocities**  
  Angular and linear velocity terms for each joint.

- **130  Body Inertias**  
  Flattened inertial properties (`cinert`) of each body element.

- **78  Body Velocities**  
  World‑frame linear and angular velocities (`cvel`) for each body.

- **23  Actuator Forces**  
  The force applied to each DoF by the combined effect of all actuators (`qfrc_actuator`).

> 🗒️ **Total:** 22 + 23 + 130 + 78 + 23 = **276** dimensions  

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🏗️ MJCF Highlights

Some notable design choices in the `model.xml` that support fast end-to-end JAX training:
- **⏱️ Timestep = 0.004**  
  A small simulation step allows for smoother dynamics and better contact stability without overwhelming compute.

- **⚡ Lightweight contact modeling**  
  Only a minimal set of contact pairs are defined (e.g., feet-to-track, some self-collisions), reducing simulation overhead and accelerating JAX-based rollouts.

> 💡 These modeling choices keep the simulation light and fast, ideal for scaling to thousands of environments in parallel.


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📊 Hyperparameters

The following settings (from `hyperparams.json`) were used to train the Sprint Humanoid with PPO:

| Parameter                 | Value        |
|---------------------------|--------------|
| Total timesteps           | 80,000,000   |
| Learning rate             | 3 × 10⁻⁴     |
| Num environments          | 4,096        |
| Unroll length             | 10 steps     |
| Batch size                | 512          |
| Episode length            | 1,000 steps  |
| Num minibatches           | 32           |
| Updates per batch         | 8            |
| Discount factor (γ)       | 0.99         |
| Entropy cost              | 0.001        |
| Max gradient norm         | 1.0          |
| PPO clipping ε            | 0.2          |
| Reward scaling            | 0.1          |
| Action repeat             | 1            |
| Normalize observations    | True         |
| RNG seed                  | 14           |
| Num evaluation rollouts   | 10           |

These are tuned for fast, stable PPO convergence

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧪 Experiment with Reward Shaping

Try tweaking the reward to something random: **the agent hops instead** or **sprints with long strides**:

```python
def _compute_reward(...):
    ...
    hopping = is_hopping()
    hop_reward = jp.where(hopping, 2.0,0.0)

    or

    foot_floor_contact = get_collision(foot_index,floor_index)
    foot_ground_penalty = jp.where(foot_floor_contact, -5.0,0.0)

    locomotion_reward += 0.5 * hop_reward + 0.5 * foot_ground_penalty
```

Or invent a race mode where finishing fast **terminates early and gives a bonus**.



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

Ready to race? This environment is a solid benchmark for dynamic locomotion and full-body coordination.

