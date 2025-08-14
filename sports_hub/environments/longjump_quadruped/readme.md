<div align="center">
  <h1>Long Jump Quadruped</h1>
</div>

<a href="https://demo.sports-hub.tech">
  <img width="1818" height="730" alt="longjump_quadruped_github" src="https://github.com/user-attachments/assets/70688817-2bda-435b-b1e7-6688bc14bce0" />
</a>

<br/>

<p align="center">
  <a href="https://demo.sports-hub.tech"><strong>🔗 Click for live demo </strong></a>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧩 Difficulty

<p align="center" style="font-size:1.1em">
⭐⭐⭐⭐☆☆☆☆☆☆<br/>
<em>4/10: Requires momentum + aerial coordination</em>
</p>

This task challenges the agent to execute a fast, well-timed takeoff and control its body mid-air to optimize landing distance. It requires fine-grained coordination between velocity generation, precise jump timing, and in-flight body posture.

⚠️ The quadruped model has a very low density making this variant easier than the humanoid variant.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


## 🧮 Reward Function

Each timestep, the reward is composed of four terms:

- 🏃‍♂️ **Forward Velocity**  
  Encourages the quadruped to sprint forward along the x‑axis.

- ⛳ **Closer to Target**  
  The change in distance from the target, between timesteps (clipped between 0 and 1)  

- 🚀 **Jump Height**  
  Height of the quadruped's torso after takeoff until the jump is complete  

- 📏 **Jump Distance**  
  Final jump distance once first contact has been detected after takeoff.  

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


## 🔁 Reset Logic

Each of the **4096 parallel environments** are initialized with small uniform noise [-0.1, +0.1] applied to joint positions and velocities of the quadruped. This encourages robustness and prevents agents from overfitting to a single deterministic start state.

The episode resets if:

- The quadruped **lands beyond the jump line**, indicating the jump has completed.
- The quadruped **falls below a height threshold** before jumping (i.e. trips on the runway).
- The quadruped **drifts too far sideways** (|y| > 8.5 m), going out of bounds.

```python
def _compute_termination(self, data: mjx.Data) -> jax.Array:
    ...
    landed_over_jumpline = jp.any(agent_collision_distances > self._jumpline_pos[0])
    
    # Get the angle of the quadrupeds torso, and make sure its within the angle limit
    torso_angle = jp.ravel(data.site_xmat[self._agent_site_index])[8]
    upright = torso_angle > self._agent_healthy_torso_angle

    # Make sure agent hasn't fallen beyond recovery before taking off
    fallen = jp.logical_not(upright) 
    fallen_before_jumpline = jp.logical_and(fallen, data.qpos[self._agent_qpos_idxs][0] < self._jumpline_pos[0])
    
    # Make sure agent is not out of bounds on the y axis
    too_far_wide = jp.abs(data.qpos[self._agent_qpos_idxs][1]) > 8.5
    
    reset_condition = jp.logical_or(fallen_before_jumpline, too_far_wide)
    reset_condition = jp.logical_or(reset_condition, landed_over_jumpline)
    return jp.where(reset_condition, 1.0, 0.0)
```

> In evaluation mode, the environment doesn't reset, and when using inference, the model simulates extra contacts for realism.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 👁 Observation Function

The agent receives a **31‑dimensional** observation vector:

- **13  Joint Positions**   
  Local joint angles, root height and orientation (quat)

- **14  Joint Velocities**  
  Angular and linear velocity terms for each joint.

- **3  Local Target Vector**  
  Position of landing target relative to the quadruped, expressed in the quadruped’s frame.

- **1 Distance To Jump Line**  
  The distance along the x-axis between the quadruped and the jumpline

> 🗒️ **Total:** 13 + 14 + 3 + 1 = **31** dimensions

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🏗️ MJCF Highlights

Some notable design choices in the `model.xml` that support fast end-to-end JAX training:
- **⏱️ Timestep = 0.003**  
  A small simulation step allows for smoother dynamics and better contact stability without overwhelming compute.

- **⚡ Lightweight contact modeling**  
  Only a minimal set of contact pairs are defined (e.g., feet-to-ground, some self-collisions), reducing simulation overhead and accelerating JAX-based rollouts.

> 💡 These modeling choices keep the simulation light and fast, ideal for scaling to thousands of environments in parallel.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📊 Hyperparameters

The following settings (from `hyperparams.json`) were used to train the Long jump Quadruped with PPO:

| Parameter                 | Value        |
|---------------------------|--------------|
| Total timesteps           | 100,000,000   |
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
| Num evaluation rollouts   | 5           |
| **Policy network**        | [512, 256, 128]     |

> 🧠 The policy network uses a deeper architecture: 3 layers with 512, 256, and 128 hidden units. This helps the agent understand this more complex environment.

These are tuned for fast, stable PPO convergence

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 🧪 Experiment with the hyperparameters

Try experimenting with a combination of different unroll lengths, total timesteps and number of environments

```xml
  "ppo_train_params": {
    "num_timesteps": 400000000,
    "num_envs": 8092,
    "unroll_length": 25,
    "batch_size": 256,
    ...
  },
  ...
}

```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

Ready to perform the Long Jump? This environment is a solid benchmark for dynamic locomotion and full-body coordination.

⚠️ Disclaimer: The reward function used here is heavily inspired by [SMPLOlympics](https://github.com/SMPLOlympics/SMPLOlympics/blob/9bba1fc90b54c02a43fd18b8629b38ed6907f4d2/phc/env/tasks/humanoid_longjump.py#L196C1-L227C1) pure RL attempt at the long jump
