<div align="center">
  <h1>Fetch Humanoid</h1>
</div>

<img width="1818" height="730" alt="humanoid_fetch" src="https://github.com/user-attachments/assets/8e763bb9-3372-41b8-b0c8-f710de717687" />

<br/>

<p align="center">
  <a href="https://i1cps.github.io/Sports-Hub-Website/"><strong>🔗 Click for live demo </strong></a>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📦 Goal

In **Humanoid Fetch**, the humanoid must fetch a ball from a random location and bring it to the center of the map

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧩 Difficulty

<p align="center" style="font-size:1.1em">
⭐⭐⭐⭐⭐⭐⭐⭐⭐☆<br/>
<em>9/10: Requires bipedal naviation and object manipulation</em>
</p>

This is a loco-manipulation environment, meaning the agent must simultaneously perform full-body locomotion (balance and movement) while manipulating the ball toward the target using its feet.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧮 Reward Function

Each timestep the agent receives a reward composed of three terms:

- 🏃‍♂️ **Speed of agent towards the ball**  
  Encourages the humanoid to constantly move towards the position of the ball

- ⛳️ **Speed of ball towards the target**  
  This reward is based on the speed of the ball towards the target, this means the humanoid should constantly be trying to move the ball towards the target as fast of possible

- ⚽ **Global speed of the ball**  
  This reward is based on the speed of the ball towards the target, this means the humanoid should constantly be trying to move the ball towards the target as fast of possible

- ❤️ **Healthy Bonus**  
  If the humanoid's torso is above 1.0m it receives a healthy reward, otherwise it receives an unhealthy penalty.

- ⚙️ **Control Cost**  
  Penalizes large actuator torques to promote efficient, smooth motions.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🔁 Reset Logic

Each of the 4096 parallel environments is initialized with uniform noise in the range [-0.01, +0.01] applied to the humanoid and ball joint positions and velocities. This promotes generalization and prevents overfitting to a fixed start state.

The episode resets if:

- The humanoid falls below a height threshold, indicating loss of balance.
- The humanoid or the ball leaves the arena bounds.

```python

def _compute_termination(self, data: mjx.Data) -> jax.Array:
    agent_root_pos = data.site_xpos[self._agent_site_index]
    global_ball_pos = data.site_xpos[self._ball_site_index]

    ball_out_bounds = jp.logical_or(
        jp.abs(global_ball_pos[0]) > self._arena_size + 1,
        jp.abs(global_ball_pos[1]) > self._arena_size + 1,
    )

    agent_out_bounds = jp.logical_or(
        jp.abs(agent_root_pos[0]) > self._arena_size + 3,
        jp.abs(agent_root_pos[1]) > self._arena_size + 3,
    )

    falling = agent_root_pos[2] < self._agent_healthy_height - 0.2

    # Reset when either; Ball is out of bounds, agent is out of bounds or the agent is below a certain height
    reset_condition = jp.logical_or(agent_out_bounds, ball_out_bounds)
    reset_condition = jp.logical_or(reset_condition, falling)
    return jp.where(reset_condition, 1.0, 0.0)
```

> During evaluation, the episode can also end when the ball reaches the target.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 👁 Observation Function

The agent receives a **288‑dimensional** observation vector:

- **22 Joint Positions**  
  Local joint angles, root height, and orientation (quat)

- **23  Joint Velocities**  
  Angular and linear velocity terms for each joint.

- **130  Body Inertias**  
  Flattened inertial properties (`cinert`) of each body element.

- **78  Body Velocities**  
  World‑frame linear and angular velocities (`cvel`) for each body.

- **23  Actuator Forces**  
  The force applied to each DoF by the combined effect of all actuators (`qfrc_actuator`).

- **3 Ball Position**  
  The ball's position translated to the humanoid frame.

- **3 Ball Linear velocity**  
  The ball's linear velocity translated to the humanoid frame.

- **3 Ball Rotational Velocity**  
  The ball's rotational velocity translated to the humanoid frame.

- **3 Target Position**  
  Target position translated to the humanoid frame.

> 🗒️ **Total:** 22 + 23 + 130 + 78 + 23 + 3 + 3 + 3 + 3 = **288** dimensions

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🏗️ MJCF Highlights

Some notable design choices in the `model.xml`:

- ⚽ **Mass** is **0.40 kg**, matching a standard football.
- ⚙️ **Friction** `(0.7, 0.7, 0.005, 0.005, 0.005)` enables realistic rolling with low resistance to spin and high resistance to slide.
- 🧩 **`solref="-5000 -30"`** configures **stiff, fast-resolving contacts**, making ball-foot interactions snappy and minimizes unnatural penetration or lag when kicked.

> 💡 Only the ball has defined contacts with the walls, the agent can pass through them, this design choice drastically speeds up agent learning, especially given the walls are only introduced to stop the ball from rolling too far

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📊 Hyperparameters

The following settings (from `hyperparams.json`) were used to train the Fetch Humanoid with PPO:

| Parameter               | Value           |
| ----------------------- | --------------- |
| Total timesteps         | 520,000,000     |
| Learning rate           | 0.5 × 10⁻⁴      |
| Num environments        | 4096            |
| Unroll length           | 10 steps        |
| Batch size              | 512             |
| Episode length          | 1000 steps      |
| Num minibatches         | 32              |
| Updates per batch       | 8               |
| Discount factor (γ)     | 0.99            |
| Entropy cost            | 0.01            |
| Max gradient norm       | 1.0             |
| PPO clipping ε          | 0.2             |
| Reward scaling          | 0.1             |
| Action repeat           | 1               |
| Normalize observations  | True            |
| RNG seed                | 0               |
| Num evaluation rollouts | 15              |
| Return best params      | True            |
| **Policy network**      | [512, 256, 128] |
| **Value network**       | [512, 256, 128] |

> 🧠 The policy and value networks use a deeper architecture: 3 layers with 512, 256, and 128 hidden units. This helps the agent understand this more complex environment.

These are tuned for fast, stable PPO convergence

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

Ready to try Fetch Humanoid? This is the most difficult task in the current collection of environments
