import jax
import mujoco
from jax import numpy as jp
from brax.io import mjcf
from mujoco import mjx
from brax.envs.base import State, PipelineEnv
from typing import Any, Tuple


# Environment to train the Humanoid Agent to long jump
class HumanoidLongJump(PipelineEnv):
    def __init__(self, sys=None):
        # Grab xml model of the scene and load into mujoco
        scene_path = "sports_hub/environments/longjump_humanoid/model.xml"
        mj_model = mujoco.MjModel.from_xml_path(scene_path)

        if not sys:
            sys = mjcf.load_model(mj_model)

        super().__init__(sys, n_frames=10, backend="mjx")

        # Meta data used for rendering in the backend
        self.render_width = mj_model.vis.global_.offwidth
        self.render_height = mj_model.vis.global_.offheight

        # Environment constants
        self._reset_noise_scale = 0.01
        self._agent_healthy_height = 1.0
        self._agent_site_id = mj_model.site("humanoid").id
        self._jumpline_pos = mj_model.geom("jump_line").pos
        self._landing_target_pos = jp.array([20, 0, 3]) 

        # Geoms
        self._sand_pit_geom_id = mj_model.geom("sand").id
        self._floor_geom_id = mj_model.geom("floor").id

        agent_collision_geom_names = ["humanoid_left_foot", "humanoid_right_foot"]
        agent_collision_geom_idxs = [mj_model.geom(name).id for name in agent_collision_geom_names]
        self._agent_collision_geom_idxs = jp.array(agent_collision_geom_idxs)

        # Reward weights
        self._agent_velocity_reward_weight = 0.01
        self._agent_jump_height_reward_weight = 1.0
        self._agent_jump_length_reward_weight = 30.0
        self._agent_closer_to_target_reward_weight = 1.0

        agent_qpos_idxs = []
        agent_qvel_idxs = []
        agent_qfrc_actuator_idxs = []
        agent_body_idxs = []

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
            if dof_id in agent_qpos_idxs:
                agent_qfrc_actuator_idxs.append(dof_id)

        # Convert the lists to JAX arrays
        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_qfrc_actuator_idxs = jp.array(agent_qfrc_actuator_idxs)
        self._agent_body_idxs = jp.array(agent_body_idxs)

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Create initial joint position and velocities with added noise
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        # Initialise Brax pipeline
        data = self.pipeline_init(qpos, qvel)

        # Get observation
        obs = self._get_obs(data)
        reward, done, zero = jp.zeros(3)
        info = {"jump_complete": zero}
        metrics = {"reward_alive": zero}
        return State(data, obs, reward, done, metrics, info)

    # Runs one timestep of the environment's dynamics.
    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        done = self._compute_termination(data=data)
        reward = self._compute_reward(data0=data0, data=data, info=state.info)
        obs = self._get_obs(data)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _compute_termination(self, data: mjx.Data) -> jax.Array:
        # Create a Lambda function to quickly check which geoms in the agent have collided with the floor or sand
        check_agent_geom_collisions = lambda target_geom_id: jax.vmap(
            lambda agent_geoms: self._geoms_colliding(data, agent_geoms, target_geom_id)
        )(self._agent_collision_geom_idxs)

        agent_collisions_with_floor = check_agent_geom_collisions(self._floor_geom_id)
        agent_collisions_with_sand = check_agent_geom_collisions(self._sand_pit_geom_id)
        agent_collisions = jp.logical_or(
            agent_collisions_with_floor, agent_collisions_with_sand
        )

        agent_geom_x_positions = data.geom_xpos[self._agent_collision_geom_idxs, 0]
        agent_collision_distances = jp.where(
            agent_collisions, agent_geom_x_positions, -jp.inf
        )
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

    def _compute_reward(self, data0: mjx.Data, data: mjx.Data, info) -> jp.ndarray:
        del info
        # Calculate current and previous pose of agent
        agent_pose_curr = data.qpos[self._agent_qpos_idxs][0:3]
        agent_pose_prev = data0.qpos[self._agent_qpos_idxs][0:3]

        # Reward for getting closer to the target position
        previous_distance = jp.linalg.norm(agent_pose_prev - self._landing_target_pos)
        current_distance = jp.linalg.norm(agent_pose_curr - self._landing_target_pos)
        change_in_distance = jp.clip(previous_distance - current_distance, 0.0, 1.0)
        closer_to_target_reward = (
            change_in_distance * self._agent_closer_to_target_reward_weight
        )

        # Calculate velocity using change in distance over time
        velocity = (
            data.site_xpos[self._agent_site_id] - data0.site_xpos[self._agent_site_id]
        ) / self.dt
        velocity_reward = velocity[0] * self._agent_velocity_reward_weight

        # Jump height reward, measured when past the jump line
        past_jumpline = data.qpos[self._agent_qpos_idxs][0] > self._jumpline_pos[0]
        jump_height = jp.where(past_jumpline, data.qpos[self._agent_qpos_idxs][2], 0.0)
        jump_height_reward = jump_height * self._agent_jump_height_reward_weight

        agent_geom_x_positions = data.geom_xpos[self._agent_collision_geom_idxs, 0]

        # Create a Lambda function to quickly check which geoms in the agent have collided with the floor and sand
        check_agent_geom_collisions = lambda target_geom_id: jax.vmap(
            lambda agent_geoms: self._geoms_colliding(data, agent_geoms, target_geom_id)
        )(self._agent_collision_geom_idxs)

        agent_collisions_with_floor = check_agent_geom_collisions(self._floor_geom_id)
        agent_collisions_with_sand = check_agent_geom_collisions(self._sand_pit_geom_id)
        agent_collisions = jp.logical_or(
            agent_collisions_with_floor, agent_collisions_with_sand
        )

        # Calculate the distance between collision geoms and jumpline to use as a reward metric
        agent_collision_distances = jp.where(
            agent_collisions, agent_geom_x_positions - self._jumpline_pos[0], -jp.inf
        )

        landed_over_jumpline = jp.any(agent_collision_distances > 0.0)

        # Jump distance is biggest distance out of all the geoms that contact the ground, (assuming they are over the jumpline) Else just 0.0
        jump_distance = jp.where(
            landed_over_jumpline, jp.max(agent_collision_distances, initial=0.0), 0.0
        )
        jump_length_reward = jump_distance * self._agent_jump_length_reward_weight

        reward = (
            closer_to_target_reward
            + velocity_reward
            + jump_height_reward
            + jump_length_reward
        )

        return reward

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        # Joint positions (Excluding the global x and y)
        # Joint velocities
        # interia values on each body
        # global velocites of each body
        # force applied to each DoF by the combined effect of all actuators
        # position of target in agent local frame
        # difference between agent and jumpline on the x axis

        agent_qpos = data.qpos[self._agent_qpos_idxs][2:]
        agent_qvel = data.qvel[self._agent_qvel_idxs]
        agent_cinert = data.cinert[self._agent_body_idxs].ravel()
        agent_cvel = data.cvel[self._agent_body_idxs].ravel()
        agent_qfrc_actuator = data.qfrc_actuator[self._agent_qfrc_actuator_idxs]

        # Get agent rotation matrix (from world to local agent frame)
        agent_humanoid_frame = data.site_xmat[self._agent_site_id]

        # Get global pose of humanoid
        agent_pose = data.site_xpos[self._agent_site_id]

        # Get target position relative to agent in world frame
        target_pos_world = self._landing_target_pos - agent_pose

        # Transform target position to local agent frame
        target_pos_local = agent_humanoid_frame.T @ target_pos_world

        jumpstart_diff = jp.array([self._jumpline_pos[0] - agent_pose[0]])

        return jp.concatenate(
            [
                agent_qpos,  # 22
                agent_qvel,  # 23
                agent_cinert,  # 130
                agent_cvel,  # 78
                agent_qfrc_actuator,  # 23
                target_pos_local,  # 3
                jumpstart_diff,  # 1
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


# Evalulation environment to train humanoid to long jump
class HumanoidLongJump_Eval(HumanoidLongJump):
    def __init__(self):
        super().__init__()

    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        reward = self._compute_reward(data0=data0, data=data, info=state.info)
        state.info["jump_complete"] = jp.logical_or(
            state.info["jump_complete"], reward > 0.0
        )
        done = 0.0
        obs = self._get_obs(data)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _compute_reward(self, data0: mjx.Data, data: mjx.Data, info):
        del data0

        agent_geom_x_positions = data.geom_xpos[self._agent_collision_geom_idxs, 0]

        # Create a Lambda function to quickly check which geoms in the agent have collided with the floor and sand
        check_agent_geom_collisions = lambda target_geom_id: jax.vmap(
            lambda agent_geoms: self._geoms_colliding(data, agent_geoms, target_geom_id)
        )(self._agent_collision_geom_idxs)

        agent_collisions_with_floor = check_agent_geom_collisions(self._floor_geom_id)
        agent_collisions_with_sand = check_agent_geom_collisions(self._sand_pit_geom_id)
        agent_collisions = jp.logical_or(
            agent_collisions_with_floor, agent_collisions_with_sand
        )

        # Calculate the distance between collision geoms and jumpline to use as a reward metric
        agent_collision_distances = jp.where(
            agent_collisions, agent_geom_x_positions - self._jumpline_pos[0], -jp.inf
        )

        landed_over_jumpline = jp.any(agent_collision_distances > 0.0)

        # Jump distance is biggest distance out of all the geoms that contact the ground, (assuming they are over the jumpline) Else just 0.0
        jump_distance = jp.where(
            landed_over_jumpline, jp.max(agent_collision_distances, initial=0.0), 0.0
        )

        jump_length_reward = jump_distance

        return jump_length_reward * (1 - info["jump_complete"])
