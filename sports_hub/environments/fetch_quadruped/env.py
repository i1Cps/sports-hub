import jax
import mujoco
from jax import numpy as jp

from brax.io import mjcf
from brax.envs.base import State, PipelineEnv
from mujoco import mjx


# Environment to train a agent Quadruped to fetch a ball and bring it to the target
class QuadrupedFetch(PipelineEnv):
    def __init__(self, sys=None):
        # Grab xml model of the scene and load into mujoco
        scene_path = "sports_hub/environments/fetch_quadruped/model.xml"
        mj_model = mujoco.MjModel.from_xml_path(scene_path)
        if not sys:
            sys = mjcf.load_model(mj_model)

        super().__init__(sys, n_frames=5, backend="mjx")

        # Meta data used for rendering in the backend
        self.render_width = mj_model.vis.global_.offwidth
        self.render_height = mj_model.vis.global_.offheight

        # Environment constants
        self._reset_noise_scale = 0.1
        self._arena_size = mj_model.geom("wall_px").size[1] - 2  # 15
        self._start_pose = mj_model.keyframe("init").qpos

        # Reward weights
        self._agent_velocity_reward_weight = 1.25
        self._agent_healthy_reward = 1
        self._agent_healthy_torso_angle = 0.7
        self._agent_ctrl_cost_weight = -0.5

        self._ball_near_target_reward_weight = 5
        self._ball_velocity_reward_weight = 2.5
        self._ball_speed_reward_weight = 0.1
        self._ball_to_target_threshold = mj_model.site("target_threshold").size[0]

        # Sites
        self._agent_site_index = mj_model.site("quadruped").id
        self._ball_site_index = mj_model.site("ball").id
        self._target_site_index = mj_model.site("target").id

        agent_qpos_idxs = []
        agent_qvel_idxs = []
        agent_body_idxs = []
        ball_qpos_idxs = []
        ball_qvel_idxs = []

        curr_qpos_idx = 0
        curr_qvel_idx = 0

        # Loop through each joint in the world model
        for i in range(mj_model.njnt):
            joint_name = mj_model.joint(i).name

            # Check if it's a root joint
            if "root" in joint_name:
                if "quadruped" in joint_name:
                    # Root joint of the current agent
                    agent_qpos_idxs.extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    agent_qvel_idxs.extend(range(curr_qvel_idx, curr_qvel_idx + 6))
                elif "ball" in joint_name:
                    # Root joint of the ball
                    ball_qpos_idxs.extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    ball_qvel_idxs.extend(range(curr_qvel_idx, curr_qvel_idx + 6))

                # Increment the index counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_idx += 7
                curr_qvel_idx += 6
                # continue

            # Include only joints starting with the agent prefix
            else:
                if "quadruped" in joint_name:
                    # Joint of current agent
                    agent_qpos_idxs.append(curr_qpos_idx)
                    agent_qvel_idxs.append(curr_qvel_idx)

                # Increment the index counters (1DOF)
                curr_qpos_idx += 1
                curr_qvel_idx += 1

        # Loop through each body in the world model
        for body_id in range(mj_model.nbody):
            body_name = mj_model.body(body_id).name

            # Store the body indices of the agent (for Observation Space)
            if "quadruped" in body_name:
                agent_body_idxs.append(body_id)
  

        # Pre compute the indices of all joints before training
        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_body_idxs = jp.array(agent_body_idxs)
        self._ball_qpos_idxs = jp.array(ball_qpos_idxs)
        self._ball_qvel_idxs = jp.array(ball_qvel_idxs)

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> State:
        # Split the rng's
        rng, rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(rng, 7)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        padding = 2
        spawn_radius = self._arena_size - padding

        # Generate random X, Y, and orientation values for the agent
        agent_pos_xy = jax.random.uniform(rng1, (2,), minval=-spawn_radius, maxval=spawn_radius)
        agent_pos_z = self._start_pose[self._agent_qpos_idxs][2]
        agent_pos = jp.array([agent_pos_xy[0],agent_pos_xy[1],agent_pos_z])

        azimuth = jax.random.uniform(rng2, (1,), minval=0, maxval=2 * jp.pi)[0]
        agent_orientation = jp.array([jp.cos(azimuth / 2), 0, 0, jp.sin(azimuth / 2)])

        # Grab specific joint positions for the agent
        agent_qpos_positions = self._start_pose[self._agent_qpos_idxs][7:]

        # Generate joint position noise for agent
        agent_qpos_noise = jax.random.uniform(rng3, agent_qpos_positions.shape, minval=low, maxval=hi)

        # Generate joint velocity noise for agent
        agent_qvel_noise = jax.random.uniform(rng4, self._agent_qvel_idxs[6:].shape, minval=low,maxval=hi)

        # Generate random X,Y, orientation and velocity values for the ball
        ball_pos_xy = jax.random.uniform(rng5, (2,), minval=-spawn_radius, maxval=spawn_radius)
        ball_pos_z = 1
        ball_pos = jp.array([ball_pos_xy[0], ball_pos_xy[1], ball_pos_z])

        ball_orientation = jp.array([1, 0, 0, 0])

        ball_vel = jax.random.uniform(rng6, (6,), minval=-1, maxval=1)

        qpos = jp.concatenate(
            [
                agent_pos,
                agent_orientation,
                agent_qpos_positions + agent_qpos_noise,
                ball_pos,
                ball_orientation,
            ]
        )

        qvel = jp.concatenate([jp.zeros(6),agent_qvel_noise,ball_vel])

        # Initialise Brax pipeline
        data = self.pipeline_init(qpos, qvel)

        # Get observation
        obs = self._get_obs(data)
        reward, done, zero = jp.zeros(3)
        metrics = {"reward_alive": zero}
        return State(data, obs, reward, done, metrics)

    # Runs one (timestep * n_frames) of the environment's dynamics.
    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        done = self._compute_termination(data=data)
        reward = self._compute_reward(data0=data0, data=data, action=action)
        obs = self._get_obs(data)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

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

    def _compute_reward(self, data0: mjx.Data, data: mjx.Data, action: jp.ndarray):
        # This calculates the velocity of the agent in the direction of the ball
        agent_velocity_towards_ball= (self._agent_to_ball_distance(data0) - self._agent_to_ball_distance(data)) / self.dt

        # This calculate the velocity of the ball in the direction of the target
        ball_velocity_towards_goal = (self._ball_to_target_distance(data0) - self._ball_to_target_distance(data)) / self.dt
        
        # We add a control cost to encourage efficient actuator control
        ctrl_cost = self._agent_ctrl_cost_weight * jp.sum(jp.square(action))

        locomotion_reward = (
            self._agent_healthy_reward
            + self._agent_velocity_reward_weight * agent_velocity_towards_ball
            + ctrl_cost
        )

        # When the ball is NOT near the target, the agent recieves a reward based on the balls velocity in the direction of the target. When the ball is near the target, it recieves a flat sustain reward
        task_reward = jp.where(
            self._ball_to_target_distance(data) < self._ball_to_target_threshold,
            self._ball_near_target_reward_weight,
            self._ball_velocity_reward_weight* ball_velocity_towards_goal,
        )

        return locomotion_reward + task_reward

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:

        # Get agent rotation matrix (from world to local agent frame)
        agent_frame = data.site_xmat[self._agent_site_index]

        # Get agent joint positions and velocities excluding the global x and y positions
        agent_qpos = data.qpos[self._agent_qpos_idxs][2:]
        agent_qvel = data.qvel[self._agent_qvel_idxs]

        agent_cfrc_ext = data.cfrc_ext[self._agent_body_idxs].ravel().clip(min=-1.0, max=1.0)

        # Get ball position relative to agent in world frame, then convert to agent frame
        ball_pos_world = data.site_xpos[self._ball_site_index] - data.site_xpos[self._agent_site_index]
        ball_pos_local = agent_frame.T @ ball_pos_world

        # Get ball velocities relative to agent in world frame, then convert to agent frame
        ball_vel_world = data.qvel[self._ball_qvel_idxs][:3] - data.qvel[self._agent_qvel_idxs][:3]
        ball_vel_local = agent_frame.T @ ball_vel_world

        ball_rotational_vel_world = data.qvel[self._ball_qvel_idxs][3:]
        ball_rotational_vel_local = agent_frame.T @ ball_rotational_vel_world

        # Get target position relative to agent in world frame
        target_pos_world = (
            data.site_xpos[self._target_site_index]
            - data.site_xpos[self._agent_site_index]
        )

        # Transform target position to local agent frame
        target_pos_local = agent_frame.T @ target_pos_world

        return jp.concatenate(
            [
                agent_qpos,                 # 13
                agent_qvel,                 # 14
                agent_cfrc_ext,             # 78
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


class QuadrupedFetch_Eval(QuadrupedFetch):
    def __init__(self):
        super().__init__()

    def _compute_termination(self, data: mjx.Data):
        global_ball_pos = data.site_xpos[self._ball_site_index]

        # Detect flipping by checking the z-component of the torso's orientation
        torso_angle = jp.ravel(data.site_xmat[self._agent_site_index])[8]
        upright = torso_angle > 0.3

        ball_reached_target = jp.linalg.norm(global_ball_pos[:2]) < self._ball_to_target_threshold

        # Reset when either; Ball is out of bounds, agent is out of bounds or the agent is about to flip over
        reset = jp.logical_or(ball_reached_target, jp.logical_not(upright))
        return jp.where(reset, 1.0, 0.0)
