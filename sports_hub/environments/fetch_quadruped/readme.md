<div align="center">
  <h1>Fetch Quadruped</h1>
</div>

<a href="https://demo.sports-hub.tech">
  <img width="1818" height="730" alt="fetch_quadruped_github" src="https://github.com/user-attachments/assets/c92dd204-8aa8-49f8-83d6-c022cb5fa2f4" />
</a>

<br/>

<p align="center">
  <a href="https://demo.sports-hub.tech"><strong>🔗 Click for live demo </strong></a>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧩 Difficulty

<p align="center" style="font-size:1.1em">
⭐⭐⭐⭐⭐⭐⭐☆☆☆<br/>
<em>7/10: Requires bipedal naviation and object manipulation</em>
</p>

This is a loco-manipulation environment, meaning the agent must simultaneously perform full-body locomotion (balance and movement) while manipulating the ball toward the target using its feet.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧮 Reward Function

Each timestep the agent receives a reward composed of five terms:

- 🏃‍♂️ **Speed of agent towards the ball**  
  Encourages the quadruped to constantly move towards the position of the ball

- ⛳️ **Speed of ball towards the target**  
  This reward is based on the speed of the ball towards the target, this means the quadruped should constantly be trying to move the ball towards the target as fast of possible

- ⚽ **Global speed of the ball**  
  This reward is based on the speed of the ball towards the target, this means the quadruped should constantly be trying to move the ball towards the target as fast of possible

- ❤️ **Alive Bonus**  
  A constant bonus awarded per timestep.

- ⚙️ **Control Cost**  
  Penalizes large actuator torques to promote efficient, smooth motions.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🔁 Reset Logic

Each of the **8192 parallel environments** is initialized with uniform noise in the range [-0.1, +0.1] applied to the quadruped and ball joint positions and velocities. This promotes generalization and prevents overfitting to a fixed start state.

The episode resets if:

- The quadruped's \*\*torso tilts beyond a certain angle threshold relative to the z-axis (vertical axis)
- The quadruped or the ball leaves the arena bounds.

```python
def _compute_termination(self, data: mjx.Data):
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

    # Get the angle of the quadrupeds torso, and make sure its within the angle limit
    torso_angle = jp.ravel(data.site_xmat[self._agent_site_index])[8]
    upright = torso_angle > self._agent_healthy_torso_angle

    # Reset when either; Ball is out of bounds, agent is out of bounds or the agent is about to flip over
    reset_condition = jp.logical_or(agent_out_bounds, ball_out_bounds)
    reset_condition = jp.logical_or(reset_condition, jp.logical_not(upright))

    return jp.where(reset_condition, 1.0, 0.0)
```

> During evaluation, the episode can also end when the ball reaches the target.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 👁 Observation Function

The agent receives a **117‑dimensional** observation vector:

- **13 Joint Positions**  
  Local joint angles, root height, and orientation (quat)

- **14  Joint Velocities**  
  Angular and linear velocity terms for each joint.

- **78 External Forces**  
  External forces acted on each body of the quadruped

- **3 Ball Position**   
  The ball's position translated to the quadruped frame.

- **3 Ball Linear velocity**   
  The ball's linear velocity translated to the quadruped frame.

- **3 Ball Rotational Velocity**   
  The ball's rotational velocity translated to the quadruped frame.

- **3 Target Position**   
  Target position translated to the quadruped frame.

> 🗒️ **Total:** 13 + 14 + 78 + 3 + 3 + 3 + 3 = **117** dimensions

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🏗️ MJCF Highlights

Some notable design choices in the `model.xml`:

- ⚽ **Density** is **5 SI**, matching the density of the modelled quadruped from Brax.
- ⚙️ **Friction** `(0.7, 0.7, 0.005, 0.005, 0.005)` enables realistic rolling with low resistance to spin and high resistance to slide.
- 🧩 **`solref="-10000 -30"`** configures **stiff, fast-resolving contacts**, making ball-foot interactions snappy and minimizes unnatural penetration or lag when kicked.

> 💡 Only the ball has defined contacts with the walls, the agent can pass through them, this design choice drastically speeds up agent learning, especially given the walls are only introduced to stop the ball from rolling too far

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📊 Hyperparameters

The following settings (from `hyperparams.json`) were used to train the Fetch Quadruped with PPO:

| Parameter               | Value                |
| ----------------------- | -------------------- |
| Total timesteps         | 350,000,000          |
| Learning rate           | 1 × 10⁻⁴             |
| Num environments        | 8192                 |
| Unroll length           | 25 steps             |
| Batch size              | 256                  |
| Episode length          | 2000 steps           |
| Num minibatches         | 64                   |
| Updates per batch       | 8                    |
| Discount factor (γ)     | 0.99                 |
| Entropy cost            | 0.01                 |
| Max gradient norm       | 1.0                  |
| PPO clipping ε          | 0.2                  |
| Reward scaling          | 0.01                 |
| Action repeat           | 1                    |
| Normalize observations  | True                 |
| RNG seed                | 70                   |
| Num evaluation rollouts | 15                    |
| **Policy network**      | [512, 256, 128, 128] |

> 🧠 The policy and value networks use a deeper architecture: 4 layers with 512, 256, 128, and 128 hidden units. This helps the agent understand this more complex environment.

These are tuned for fast, stable PPO convergence

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

Ready to try Fetch Quadruped?
