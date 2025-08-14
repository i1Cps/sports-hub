import jax
import mujoco
from jax import numpy as jp
from brax.io import mjcf
from brax.envs.base import State, PipelineEnv
from typing import Any, Tuple
from mujoco import mjx

# Environment to train the Humanoid agent to sprint across hurdles
class HumanoidHurdles(PipelineEnv):
    def __init__(self, sys=None):
        # Grab xml model of the scene and load into mujoco
        scene_path = "sports_hub/environments/hurdles_humanoid/model.xml"
        mj_model = mujoco.MjModel.from_xml_path(scene_path)
        if not sys:
            sys = mjcf.load_model(mj_model)
        super().__init__(sys, n_frames=5, backend="mjx")

        # Meta data used for rendering in the backend
        self.render_width = mj_model.vis.global_.offwidth
        self.render_height = mj_model.vis.global_.offheight

        # Environment constants
        self._reset_noise_scale = 0.01
        self._agent_site_index = mj_model.site("humanoid").id
        self._agent_healthy_height = 1.00
        self._start_pose = mj_model.qpos0
        self._finish_line_geom_index = mj_model.geom("finish_line").id
        self._finish_line_pos = mj_model.geom_pos[self._finish_line_geom_index]
        self._agent_collision_geom_names = [
            "humanoid_right_foot",
            "humanoid_left_foot",
            "humanoid_right_shin",
            "humanoid_left_shin",
            "humanoid_right_thigh",
            "humanoid_left_thigh",
        ]

        # Reward weights
        self._agent_velocity_reward_weight = 3.50
        self._agent_healthy_reward_weight = 2
        self._agent_ctrl_cost_weight = 0.1

        agent_qpos_idxs = []
        agent_qvel_idxs = []
        agent_qfrc_actuator_idxs = []
        agent_body_idxs = []
        agent_rangefinder_idxs = []
        agent_collision_geom_idxs = []
        hurdle_geom_idxs = []

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

        for dof_id in range(mj_model.nv):
            if dof_id in agent_qvel_idxs:
                agent_qfrc_actuator_idxs.append(dof_id)

        # Loop through sensors to get range finders
        for sensor_id in range(mj_model.nsensor):
            sensor_name = mj_model.sensor(sensor_id).name
            is_ray = "ray" in sensor_name

            if is_ray:
                agent_rangefinder_idxs.append(sensor_id)

        # Loop through each geom in the world model
        for geom_id in range(mj_model.ngeom):
            geom_name = mj_model.geom(geom_id).name
            # Store the collision geoms indices of the agent
            if geom_name in self._agent_collision_geom_names:
                agent_collision_geom_idxs.append(geom_id)
            # Store the infices of the hurdles
            if "hurdle" in geom_name:
                hurdle_geom_idxs.append(geom_id)

        # Convert the lists to JAX arrays
        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_qfrc_actuator_idxs = jp.array(agent_qfrc_actuator_idxs)
        self._agent_body_idxs = jp.array(agent_body_idxs)
        self._agent_rangefinder_idxs = jp.array(agent_rangefinder_idxs)
        self._agent_collision_geom_idxs = jp.array(agent_collision_geom_idxs)
        self._hurdle_geom_idxs = jp.array(hurdle_geom_idxs)

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> State:
        rng, rng_x, rng_y, rng_qpos_noise, rng_qvel_noise = jax.random.split(rng, 5)

        # Create initial joint position and velocities with added noise
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos_noise = jax.random.uniform(
            rng_qpos_noise, self._agent_qpos_idxs[7:].shape, minval=low, maxval=hi
        )
        start_x = jax.random.uniform(
            rng_x,
            (1,),
            minval=-1,
            maxval=1,
        )
        start_y = jax.random.uniform(rng_y, (1,), minval=-4, maxval=4)
        qpos = jp.concatenate(
            [start_x, start_y, self._start_pose[2:7], qpos_noise + self._start_pose[7:]]
        )
        qvel = jax.random.uniform(rng_qvel_noise, (self.sys.nv,), minval=low, maxval=hi)

        # Initialise Brax pipeline
        data = self.pipeline_init(qpos, qvel)

        # Get observation
        obs = self._get_obs(data)
        reward, done, zero = jp.zeros(3)
        metrics = {"velocity": zero}
        return State(data, obs, reward, done, metrics)

    # Runs one timestep of the environment's dynamics.
    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        reward, info = self._compute_reward(data0=data0, data=data, action=action)
        done = self._compute_termination(data)
        obs = self._get_obs(data)

        state = state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
        state.metrics.update(
            velocity=info["velocity"],
        )
        return state

    def _compute_termination(self, data: mjx.Data):
        # Get the height of the agent humanoids torso, make sure its above the height limit
        torso_height = data.site_xpos[self._agent_site_index][2]
        upright = torso_height > self._agent_healthy_height - 0.2

        reset_condition = jp.logical_not(upright)
        return jp.where(reset_condition, 1.0, 0.0)

    def _compute_reward(self, data0: mjx.Data, data: mjx.Data, action):
        # Calculate velocity using change in distance over time
        velocity = (data.site_xpos[self._agent_site_index] - data0.site_xpos[self._agent_site_index]) / self.dt
        velocity_reward = velocity[0] * self._agent_velocity_reward_weight

        # We add a control cost to encourage efficient actuator control
        ctrl_cost = self._agent_ctrl_cost_weight * jp.sum(jp.square(action))

        # Healthy reward for maintaining a healthy position
        healthy_reward = jp.where(
            data.qpos[self._agent_qpos_idxs][2] > self._agent_healthy_height,
            self._agent_healthy_reward_weight,
            -self._agent_healthy_reward_weight,
        )
        locomotion_reward = velocity_reward + ctrl_cost + healthy_reward

        collisions = jp.array(
            [
                self._geoms_colliding(data, hurdle, agent_collision_geom)
                for hurdle in self._hurdle_geom_idxs
                for agent_collision_geom in self._agent_collision_geom_idxs
            ]
        )

        collision_penalty = jp.where(jp.any(collisions), -9.0, 0.0)

        info = {"velocity": velocity[0]}

        total_reward = locomotion_reward + collision_penalty
        return total_reward, info

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
                data.qpos[self._agent_qpos_idxs][2:],               # 22
                data.qvel[self._agent_qvel_idxs],                   # 23
                data.cinert[self._agent_body_idxs].ravel(),         # 140
                data.cvel[self._agent_body_idxs].ravel(),           # 84
                data.qfrc_actuator[self._agent_qfrc_actuator_idxs], # 23
                data.cfrc_ext[self._agent_body_idxs]
                .ravel()
                .clip(min=-1.0, max=1.0),                           # 84
                ranges,                                             # 30
            ]
        )

    def _get_collision_info(
        self, contact: Any, geom1: int, geom2: int
    ) -> Tuple[jax.Array, jax.Array]:
        """Get the distance and normal of the collision between two geoms."""
        mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
        mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
        idx = jp.where(mask, contact.dist, 1e4).argmin()
        dist = contact.dist[idx] * mask[idx]
        normal = (dist < 0) * contact.frame[idx, 0, :3]
        return dist, normal

    def _geoms_colliding(self, state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
        """Return True if the two geoms are colliding."""
        return self._get_collision_info(state.contact, geom1, geom2)[0] < 0


class HumanoidHurdles_Eval(HumanoidHurdles):
    def __init__(self):
        super().__init__()

    def _compute_termination(self, data: mjx.Data):
        # Get the root position of the agent
        agent_root_pos = data.site_xpos[self._agent_site_index]

        # Check if agent has fallen over
        fallen = agent_root_pos[2] < 0.5

        # Check if agent has completed track
        finished_track = agent_root_pos[0] > self._finish_line_pos[0] + 1

        jax.lax.cond(
            finished_track,
            lambda: (jax.debug.print("Reached finish line"), 0)[1],
            lambda: 0,
        )

        terminate = jp.logical_or(fallen, finished_track)
        return jp.where(terminate, 1.0, 0.0)
