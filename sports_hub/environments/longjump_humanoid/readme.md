<div align="center">
  <h1>Long jump Humanoid</h1>
</div>

<a href="https://demo.sports-hub.tech">
  <img width="1818" height="730" alt="longjump_humanoid_github" src="https://github.com/user-attachments/assets/00b0d780-b7e1-4e3b-8d53-1100b60a98cf" />
</a>

<br/>

<p align="center">
  <a href="https://demo.sports-hub.tech"><strong>🔗 Click for live demo </strong></a>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧩 Difficulty

<p align="center" style="font-size:1.1em">
⭐⭐⭐⭐⭐⭐☆☆☆☆<br/>
<em>6/10: Requires momentum + aerial coordination</em>
</p>

This task challenges the agent to execute a fast, well-timed takeoff and control its body mid-air to optimize landing distance. It requires fine-grained coordination between velocity generation, precise jump timing, and in-flight body posture.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


## 🧮 Reward Function

Each timestep, the reward is composed of four terms:

- 🏃‍♂️ **Forward Velocity**  
  Encourages the humanoid to sprint forward along the x‑axis.

- ⛳ **Closer to Target**  
  The change in distance from the target, between timesteps (clipped between 0 and 1)  

- 🚀 **Jump Height**  
  Height of the humanoid's torso after takeoff until the jump is complete  

- 📏 **Jump Distance**  
  Final jump distance once first contact has been detected after takeoff.  

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


## 🔁 Reset Logic

Each of the **4096 parallel environments** are initialized with small uniform noise [-0.01, +0.01] applied to joint positions and velocities of the humanoid. This encourages robustness and prevents agents from overfitting to a single deterministic start state.

The episode resets if:

- The humanoid **lands beyond the jump line**, indicating the jump has completed.
- The humanoid **falls below a height threshold** before jumping (i.e. trips on the runway).
- The humanoid **drifts too far sideways** (|y| > 8.5 m), going out of bounds.

```python
def _compute_termination(self, data: mjx.Data) -> jax.Array:
    ...
    landed_over_jumpline = jp.any(agent_collision_distances > self._jumpline_pos[0])
    
    # Make sure agent hasn't fallen beyond recovery before taking off
    fallen = data.qpos[self._agent_qpos_idxs][2] < self._agent_healthy_height
    fallen_before_jumpline = jp.logical_and(
        fallen, data.qpos[self._agent_qpos_idxs][0] < self._jumpline_pos[0]
    )
    
    # Make sure agent is not out of bounds on the y axis
    too_far_wide = jp.abs(data.qpos[self._agent_qpos_idxs][1]) > 8.5
    
    reset_condition = jp.logical_or(fallen_before_jumpline, too_far_wide)
    reset_condition = jp.logical_or(reset_condition, landed_over_jumpline)
    return jp.where(reset_condition, 1.0, 0.0)
```

> In evaluation mode, the environment doesn't reset, and when using inference, the model simulates extra contacts for realism.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 👁 Observation Function

The agent receives a **279‑dimensional** observation vector:

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

- **3  Local Target Vector**  
  Position of landing target relative to the humanoid, expressed in the humanoid’s frame.

- **1 Distance To Jump Line**  
  The distance along the x-axis between the humanoid and the jumpline

> 🗒️ **Total:** 22 + 23 + 130 + 78 + 23 + 3 + 1 = **280** dimensions  

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🏗️ MJCF Highlights

Some notable design choices in the `model.xml` that support fast end-to-end JAX training:
- **⏱️ Timestep = 0.002**  
  A small simulation step allows for smoother dynamics and better contact stability without overwhelming compute.

- **⚡ Lightweight contact modeling**  
  Only a minimal set of contact pairs are defined (e.g., feet-to-ground, some self-collisions), reducing simulation overhead and accelerating JAX-based rollouts.

> 💡 These modeling choices keep the simulation light and fast, ideal for scaling to thousands of environments in parallel.


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📊 Hyperparameters

The following settings (from `hyperparams.json`) were used to train the Long jump Humanoid with PPO:

| Parameter                 | Value        |
|---------------------------|--------------|
| Total timesteps           | 550,000,000   |
| Learning rate             | 1 × 10⁻⁴     |
| Num environments          | 4,096        |
| Unroll length             | 10 steps     |
| Batch size                | 512          |
| Episode length            | 1,000 steps  |
| Num minibatches           | 32           |
| Updates per batch         | 8            |
| Discount factor (γ)       | 0.99         |
| Entropy cost              | 0.01        |
| Max gradient norm         | 1.0          |
| PPO clipping ε            | 0.2          |
| Reward scaling            | 0.1          |
| Action repeat             | 1            |
| Normalize observations    | True         |
| RNG seed                  | 30           |
| Num evaluation rollouts   | 8           |
| **Policy network**        | [512, 256, 128]     |

> 🧠 The policy network uses a deeper architecture: 3 layers with 512, 256, and 128 hidden units. This helps the agent understand this more complex environment.

These are tuned for fast, stable PPO convergence

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧪 Experiment with simulation settings

Try experimenting with different iteration and timestep values! 

```xml
<mujoco model="humanoid_longjump">
  <option timestep="0.004" iterations="1" ls_iterations="5" solver="Newton">
    <flag eulerdamp="disable"/>
  </option>
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

Ready to perform the Long Jump? This environment is a solid benchmark for dynamic locomotion and full-body coordination.

⚠️ Disclaimer: The reward function used here is heavily inspired by [SMPLOlympics](https://github.com/SMPLOlympics/SMPLOlympics/blob/9bba1fc90b54c02a43fd18b8629b38ed6907f4d2/phc/env/tasks/humanoid_longjump.py#L196C1-L227C1) pure RL attempt at the long jump

