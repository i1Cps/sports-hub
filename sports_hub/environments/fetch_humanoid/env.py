import jax
import mujoco
from jax import numpy as jp

from brax.io import mjcf
from brax.envs.base import State, PipelineEnv
from mujoco import mjx


# Environment to train the Humanoid agent to fetch a ball and bring it to the target
class HumanoidFetch(PipelineEnv):
    def __init__(self, sys=None):
        # Grab xml model of the scene and load into mujoco
        scene_path = "sports_hub/environments/fetch_humanoid/model.xml"
        mj_model = mujoco.MjModel.from_xml_path(scene_path)
        if sys is None:
            sys = mjcf.load_model(mj_model)
        super().__init__(sys, n_frames=5, backend="mjx")

        # Meta data used for rendering in the backend
        self.render_width = mj_model.vis.global_.offwidth
        self.render_height = mj_model.vis.global_.offheight

        # Environment constants
        self._agent_healthy_height = 1.0
        self._reset_noise_scale = 0.01
        self._start_pose = mj_model.qpos0
        self._arena_size = mj_model.geom("wall_px").size[1] - 2  # 15
        self._ball_to_target_threshold = mj_model.site("target_threshold").size[0] - mj_model.geom("ball").size[0]

        # Reward weights
        self._agent_healthy_reward_weight = 5.0
        self._agent_near_ball_reward_weight = 1
        self._agent_ctrl_cost_weight = -0.0
        self._agent_speed_towards_ball_reward_weight = 3.5
        self._ball_speed_towards_target_reward_weight = 1.5
        self._ball_speed_global_reward_weight = 1.0
        self._ball_near_target_reward_weight = 5

        # Sites
        self._agent_site_index = mj_model.site("humanoid").id
        self._ball_site_index = mj_model.site("ball").id
        self._target_site_index = mj_model.site("target").id

        agent_qpos_idxs = []
        agent_qvel_idxs = []
        agent_body_idxs = []
        agent_qfrc_actuator_idxs = []
        ball_qpos_idxs = []
        ball_qvel_idxs = []

        curr_qpos_idx = 0
        curr_qvel_idx = 0

        # Loop through each joint in the world model
        for i in range(mj_model.njnt):
            joint_name = mj_model.joint(i).name

            # Check if it's a root joint
            if "root" in joint_name:
                if "humanoid" in joint_name:
                    # Root joint of the agent
                    agent_qpos_idxs.extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    agent_qvel_idxs.extend(range(curr_qvel_idx, curr_qvel_idx + 6))
                elif "ball" in joint_name:
                    # Root joint of the ball
                    ball_qpos_idxs.extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    ball_qvel_idxs.extend(range(curr_qvel_idx, curr_qvel_idx + 6))

                # Increment the index counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_idx += 7
                curr_qvel_idx += 6

            # Include only joints starting with the agent prefix
            else:
                if "humanoid" in joint_name:
                    agent_qpos_idxs.append(curr_qpos_idx)
                    agent_qvel_idxs.append(curr_qvel_idx)

                # Increment the index counters (1DOF)
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

        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_qfrc_actuator_idxs = jp.array(agent_qfrc_actuator_idxs)
        self._agent_body_idxs = jp.array(agent_body_idxs)

        self._ball_qpos_idxs = jp.array(ball_qpos_idxs)
        self._ball_qvel_idxs = jp.array(ball_qvel_idxs)

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> State:
        # Split the rng
        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 9)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        padding = 3
        spawn_radius = self._arena_size - padding

        # Generate random X, Y, and orientation values for the agent
        agent_pos_xy = jax.random.uniform(rng1, (2,), minval=-spawn_radius, maxval=spawn_radius)
        agent_pos_z = self._start_pose[self._agent_qpos_idxs][2]
        agent_pos = jp.array([agent_pos_xy[0], agent_pos_xy[1], agent_pos_z])

        azimuth = jax.random.uniform(rng2, (1,), minval=0, maxval=2 * jp.pi)[0]
        agent_orientation = jp.array([jp.cos(azimuth / 2), 0, 0, jp.sin(azimuth / 2)])

        # Get the original joint positions
        agent_joint_pos = self._start_pose[self._agent_qpos_idxs][7:]

        # Complete the pose for the agent
        agent_qpos = jp.concatenate(
            [
                agent_pos,
                agent_orientation,
                agent_joint_pos,
            ]
        )

        # Generate random X, Y, and orientation values for the ball
        ball_pos_xy = jax.random.uniform(rng3, (2,), minval=-spawn_radius, maxval=spawn_radius)
        ball_pos_z = 0.15
        ball_pos = jp.array([ball_pos_xy[0], ball_pos_xy[1], ball_pos_z])
        ball_orientation = jp.array([1, 0, 0, 0])

        # Complete the pose for the ball
        ball_qpos = jp.concatenate([ball_pos, ball_orientation])

        # Complete the velocity for the ball
        ball_vel_xy = jax.random.uniform(rng8, (2,), minval=-1, maxval=1)
        ball_qvel = jp.concatenate([ball_vel_xy, jp.zeros(4)])

        # Generate noise
        agent_qpos_noise = jax.random.uniform(rng4, agent_qpos.shape, minval=low, maxval=hi)
        agent_qvel_noise = jax.random.uniform(rng5, self._agent_qvel_idxs.shape, minval=low, maxval=hi)
        ball_qpos_noise = jax.random.uniform(rng6, ball_qpos.shape, minval=low, maxval=hi)
        ball_qvel_noise = jax.random.uniform(rng7, self._ball_qvel_idxs.shape, minval=low,maxval=hi)

        # Add noise to everything
        agent_qpos_with_noise = agent_qpos + agent_qpos_noise
        ball_qpos_with_noise = ball_qpos + ball_qpos_noise
        ball_qvel_with_noise = ball_qvel + ball_qvel_noise

        # Create qpos and qvel empty "container"
        qpos = jp.zeros((self.sys.nq,))
        qvel = jp.zeros((self.sys.nv,))

        # Set agent and ball qpos
        qpos = qpos.at[self._agent_qpos_idxs].set(agent_qpos_with_noise)
        qpos = qpos.at[self._ball_qpos_idxs].set(ball_qpos_with_noise)

        # Set agent and ball qvel
        qvel = qvel.at[self._agent_qvel_idxs].set(agent_qvel_noise)
        qvel = qvel.at[self._ball_qvel_idxs].set(ball_qvel_with_noise)

        # Initialise Brax pipeline
        data = self.pipeline_init(qpos, qvel)

        # Get observation
        obs = self._get_obs(data)
        reward, done, zero = jp.zeros(3)
        metrics = {"reward_alive": zero}
        return State(data, obs, reward, done, metrics)

    # Runs one timestep of the environment's dynamics.
    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        done = self._compute_termination(data=data)
        reward = self._compute_reward(data0=data0, data=data, action=action)
        obs = self._get_obs(data)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

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

    def _compute_reward(self, data0: mjx.Data, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        # Calculates the speed of the agent in the direction of the ball
        agent_speed_towards_ball = (self._agent_to_ball_distance(data0) - self._agent_to_ball_distance(data)) / self.dt

        # Calculate the speed of the ball in the direction of the target
        ball_speed_towards_target = (self._ball_to_target_distance(data0) - self._ball_to_target_distance(data)) / self.dt

        velocity_of_ball = (data0.site_xpos[self._ball_site_index] - data.site_xpos[self._ball_site_index]) / self.dt
        speed_of_ball_global = jp.linalg.norm(velocity_of_ball[:2])

        # We add a control cost to encourage efficient actuator control
        ctrl_cost = self._agent_ctrl_cost_weight * jp.sum(jp.square(action))

        healthy = jp.where(
            data.site_xpos[self._agent_site_index][2] > self._agent_healthy_height,
            self._agent_healthy_reward_weight,
            -self._agent_healthy_reward_weight,
        )

        # Base locomotion reward
        base_reward = ctrl_cost + healthy

        locomotion_reward = self._agent_speed_towards_ball_reward_weight * agent_speed_towards_ball 

        # When the ball is NOT near the target, the agent recieves a reward based on the balls velocity in the direction of the target. 
        # When the ball is near the target, it recieves a flat reward
        task_reward = jp.where(
            self._ball_to_target_distance(data) < self._ball_to_target_threshold,
            self._ball_near_target_reward_weight,
            self._ball_speed_towards_target_reward_weight * ball_speed_towards_target
            + self._ball_speed_global_reward_weight * speed_of_ball_global,
        )

        return base_reward + locomotion_reward + task_reward

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        # Get agent rotation matrix (from world to local agent frame)
        agent_frame = data.site_xmat[self._agent_site_index]

        # Get agent joint positions and velocities excluding the global x and y positions
        agent_qpos = data.qpos[self._agent_qpos_idxs][2:]
        agent_qvel = data.qvel[self._agent_qvel_idxs]

        # Interia values on each body
        agent_cinert = data.cinert[self._agent_body_idxs].ravel()

        # Global velocites of each body
        agent_cvel = data.cvel[self._agent_body_idxs].ravel()

        # The force applied to each DoF by the combined effect of all actuators (qfrc_actuator).
        agent_qfrc_actuators = data.qfrc_actuator[self._agent_qfrc_actuator_idxs]

        # Get ball position relative to agent in world frame, then convert to agent frame
        ball_pos_world = data.site_xpos[self._ball_site_index] - data.site_xpos[self._agent_site_index]
        ball_pos_local = agent_frame.T @ ball_pos_world

        # Get ball velocities relative to agent in world frame, then convert to agent frame
        ball_vel_world = data.qvel[self._ball_qvel_idxs][:3] - data.qvel[self._agent_qvel_idxs][:3]
        ball_vel_local = agent_frame.T @ ball_vel_world

        ball_rotational_vel_world = data.qvel[self._ball_qvel_idxs][3:]
        ball_rotational_vel_local = agent_frame.T @ ball_rotational_vel_world

        # Get target position relative to agent in world frame
        target_pos_world = data.site_xpos[self._target_site_index] - data.site_xpos[self._agent_site_index]

        # Transform target position to local agent frame
        target_pos_local = agent_frame.T @ target_pos_world

        return jp.concatenate(
            [
                agent_qpos,                 # 22
                agent_qvel,                 # 23
                agent_cinert,               # 130
                agent_cvel,                 # 78
                agent_qfrc_actuators,       # 23
                ball_pos_local,             # 3
                ball_vel_local,             # 3
                ball_rotational_vel_local,  # 3
                target_pos_local,           # 3
            ]
        )

    # HELPER FUNCTIONS:

    # Returns distance from the agent to the ball.
    def _agent_to_ball_distance(self, data):
        self_to_ball = data.site_xpos[self._ball_site_index] - data.site_xpos[self._agent_site_index]
        return jp.linalg.norm(self_to_ball[:2])

    # Returns distance from the ball to the target.
    def _ball_to_target_distance(self, data):
        ball_to_target = data.site_xpos[self._target_site_index] - data.site_xpos[self._ball_site_index]
        return jp.linalg.norm(ball_to_target[:2])

class HumanoidFetch_Eval(HumanoidFetch):
    def __init__(self):
        super().__init__()

    def _compute_termination(self, data: mjx.Data) -> jax.Array:
        agent_root_pos = data.site_xpos[self._agent_site_index]
        global_ball_pos = data.site_xpos[self._ball_site_index]

        falling = agent_root_pos[2] < 0.5
        ball_reached_target = jp.linalg.norm(global_ball_pos[:2]) < 0.3

        jax.lax.cond(
            ball_reached_target,
            lambda: (jax.debug.print("ball reached target"), 0)[1],
            lambda: 0,
        )

        reset_condition = jp.logical_or(ball_reached_target, falling)
        return jp.where(reset_condition, 1.0, 0.0)
