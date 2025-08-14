import jax
import mujoco
from jax import numpy as jp
from brax.io import mjcf
from brax.envs.base import State, PipelineEnv
from mujoco import mjx
from typing import Tuple, Dict


# Environment to train the Humanoid agent to sprint across a platform of gaps
class HumanoidGaps(PipelineEnv):
    def __init__(self, sys=None):
        # Grab xml model of the scene and load into mujoco
        scene_path = "sports_hub/environments/gaps_humanoid/model.xml"
        mj_model = mujoco.MjModel.from_xml_path(scene_path)
        if not sys:
            sys = mjcf.load_model(mj_model)
        super().__init__(sys, n_frames=5, backend="mjx")

        # Meta data used for rendering in the backend
        self.render_width = mj_model.vis.global_.offwidth
        self.render_height = mj_model.vis.global_.offheight

        # Environment constants
        self._reset_noise_scale = 0.01
        self._finish_line_geom_index = mj_model.geom("finish_line").id
        self._finish_line_pos = mj_model.geom_pos[self._finish_line_geom_index]
        self._agent_site_index = mj_model.site("humanoid").id
        self._agent_healthy_height = 1.00

        # Reward weights
        self._agent_velocity_reward_weight = 2.55
        self._agent_healthy_reward = 2
        self._agent_ctrl_cost_weight = 0.1

        agent_qpos_idxs = []
        agent_qvel_idxs = []
        agent_qfrc_actuator_idxs = []
        agent_body_idxs = []
        agent_rangefinder_idxs = []

        curr_qpos_idx = 0
        curr_qvel_idx = 0

        # Loop through each joint in the world model
        for joint_id in range(mj_model.njnt):
            joint_name = mj_model.joint(joint_id).name

            # Check if joint is a root joint
            if "root" in joint_name:
                if "humanoid" in joint_name:
                    agent_qpos_idxs.extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    agent_qvel_idxs.extend(range(curr_qvel_idx, curr_qvel_idx + 6))

                # Increment the index counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_idx += 7
                curr_qvel_idx += 6

            # Include only joints starting with the agent prefix
            else:
                if "humanoid" in joint_name:
                    agent_qpos_idxs.append(curr_qpos_idx)
                    agent_qvel_idxs.append(curr_qvel_idx)
                curr_qpos_idx += 1
                curr_qvel_idx += 1

        # Loop through each body in the world model
        for body_id in range(mj_model.nbody):
            body_name = mj_model.body(body_id).name

            # Store the body indices of the agent (for Observation Space)
            if "humanoid" in body_name:
                agent_body_idxs.append(body_id)

        # Loop through each DOF in the world model and store the ones that belong to the agent humanoid
        for dof_id in range(mj_model.nv):
            if dof_id in agent_qvel_idxs:
                agent_qfrc_actuator_idxs.append(dof_id)

        for sensor_id in range(mj_model.nsensor):
            sensor_name = mj_model.sensor(sensor_id).name
            if "ray" in sensor_name:
                agent_rangefinder_idxs.append(sensor_id)

        # Convert the lists to JAX arrays
        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_body_idxs = jp.array(agent_body_idxs)
        self._agent_rangefinder_idxs = jp.array(agent_rangefinder_idxs)
        self._agent_qfrc_actuator_idxs = jp.array(agent_qfrc_actuator_idxs)

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Create initial joint position and velocities with added noise
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        randy = jax.random.uniform(rng1, (), minval=-3,maxval=3)
        qpos = qpos.at[self._agent_qpos_idxs[1]].set(randy)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        # Initialise Brax pipeline
        data = self.pipeline_init(qpos, qvel)

        # Get observation
        obs = self._get_obs(data)
        reward, done, zero = jp.zeros(3)
        metrics = {"velocity": zero}
        return State(data, obs, reward, done, metrics)

    # Runs one (timestep * n_frames) of the environment's dynamics.
    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        reward, info = self._compute_reward(data0=data0, data=data, action=action)
        done = self._compute_termination(data)
        obs = self._get_obs(data)

        state = state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
        state.metrics.update(velocity=info["velocity"])
        return state

    def _compute_termination(self, data: mjx.Data) -> jax.Array:
        # Get the height of the humanoids torso, make sure its above the height limit
        torso_height = data.site_xpos[self._agent_site_index][2]
        fallen = torso_height < self._agent_healthy_height

        reset_condition = fallen
        return jp.where(reset_condition, 1.0, 0.0)

    def _compute_reward(self, data0: mjx.Data, data: mjx.Data, action) ->Tuple[jp.ndarray, Dict]:
        # Calculate velocity using change in distance over time
        velocity = (data.site_xpos[self._agent_site_index] - data0.site_xpos[self._agent_site_index]) / self.dt
        velocity_reward = velocity[0] * self._agent_velocity_reward_weight

        # We add a control cost to encourage efficient actuator control
        ctrl_cost = self._agent_ctrl_cost_weight * jp.sum(jp.square(action))

        # Healthy reward for maintaining a healthy position
        locomotion_reward = velocity_reward + ctrl_cost + self._agent_healthy_reward

        info = {"velocity": velocity[0]}

        return locomotion_reward, info

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        # Joint positions
        # Joint velocities
        # interia values on each body
        # global velocites of each body
        # force applied to each DoF by the combined effect of all actuators
        # Range finders

        def _get_range_finders(data: mjx.Data):
            rf_readings = data.sensordata[self._agent_rangefinder_idxs]
            no_intersection = -1.0
            return jp.where(rf_readings == no_intersection, 1.0, jp.tanh(rf_readings))
        ranges = _get_range_finders(data)

        return jp.concatenate(
            [
                data.qpos[self._agent_qpos_idxs][2:],                   # 22
                data.qvel[self._agent_qvel_idxs],                       # 23
                data.cinert[self._agent_body_idxs].ravel(),             # 140
                data.cvel[self._agent_body_idxs].ravel(),               # 84
                data.qfrc_actuator[self._agent_qfrc_actuator_idxs],     # 23
                ranges,                                                 # 20
            ]
        )


class HumanoidGaps_Eval(HumanoidGaps):
    def __init__(self):
        super().__init__()

    def _compute_termination(self, data: mjx.Data) -> jax.Array:
        # Get the root position of the agent
        agent_root_pos = data.site_xpos[self._agent_site_index]

        # Check if agent has fallen over
        fallen = agent_root_pos[2] < 0.1

        # Check if agent has completed track
        finished_track = agent_root_pos[0] > self._finish_line_pos[0] + 1

        terminate = jp.logical_or(fallen, finished_track)
        return jp.where(terminate, 1.0, 0.0)
