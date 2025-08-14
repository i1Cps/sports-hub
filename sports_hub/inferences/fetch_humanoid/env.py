import re
from typing import Any, Tuple, List
import mujoco
import jax
from jax import numpy as jp
from mujoco import mjx
import random


# Environment to help the inference engine construct observation and evaluate the state for a control policy
class FetchHumanoid_Inference:
    def __init__(self, number_of_agents=1, debug=False):
        scene_path = "sports_hub/inferences/fetch_humanoid/base_model.xml"

        # Original spec to copy from
        original_spec = mujoco.MjSpec.from_file(scene_path)

        # Main Spec that we are buliding on
        spec = mujoco.MjSpec.from_file(scene_path)

        # Remove the original humanoid body form the base model
        spec.delete(spec.body("humanoid"))

        # Create a dummy frame in the main spec and use it to attach each new humanoid too
        dummy_frame = spec.worldbody.add_frame()

        # Pre compute colours for each humanoid and its ball
        colours = []

        for i in range(number_of_agents):
            r = random.random()
            g = random.random()
            b = random.random()
            colours.append([r, g, b, 1.0])

        for i in range(number_of_agents):
            # Get a deep copy of humanoid body
            spec_copy = original_spec.copy()
            humanoid_body_copy = spec_copy.body("humanoid")

            # Attach it to the dummy frame of the main Spec (This is the correct way to attach bodies to specs)
            dummy_frame.attach_body(humanoid_body_copy, f"{i+1}_", "")

            # Sample a material colour
            humanoid_material = spec.material(f"{i+1}_body")
            humanoid_material.rgba = colours[i]

        # Add the balls
        for i in range(number_of_agents):
            # Create a body for the ball
            ball = spec.worldbody.add_body()
            ball.name = f"{i+1}_ball"
            ball.pos = [(i + 1) * 2, (i + 1) * 2, 3]

            # Add a free joint to it, so it can roll around
            ball.add_freejoint(name=f"{i+1}_ball_root")

            # Sort out the material colour using the original ball texture from the base model xml file
            ball_material = spec.add_material(name=f"{i+1}_ball", rgba=colours[i])
            texture_index = mujoco.mjtTextureRole.mjTEXROLE_RGB
            ball_material.textures[texture_index] = "football"

            # Create the ball geom
            ball_geom = ball.add_geom(
                name=f"{i+1}_ball",
                size=[0.15, 0, 0],
                condim=6,
                material=f"{i+1}_ball",
                mass=0.40
            )

            # Create the ball site
            ball.add_site(name=f"{i+1}_ball", size=[0.01, 0, 0], pos=[0, 0, 0])

            # Contacts time ...
            spec.add_pair(
                geomname1=f"{i+1}_humanoid_left_foot",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -30],
                
            )
            spec.add_pair(
                geomname1=f"{i+1}_humanoid_right_foot",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -30],
            )
            spec.add_pair(
                geomname1=f"{i+1}_humanoid_left_shin",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -30],
            )
            spec.add_pair(
                geomname1=f"{i+1}_humanoid_right_shin",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -30],
            )

            spec.add_pair(
                geomname1="floor",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -30],
            )

            spec.add_pair(
                geomname1="wall_nx",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -80],
            )

            spec.add_pair(
                geomname1="wall_ny",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -80],
            )

            spec.add_pair(
                geomname1="wall_px",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -80],
            )

            spec.add_pair(
                geomname1="wall_py",
                geomname2=ball_geom.name,
                condim=6,
                friction=[0.7,0.7, 0.005, 0.005, 0.005],
                solref=[-5000, -80],
            )

        # Compile and print the new xml file to proof read
        model = spec.compile()
        self.model = model
        self.debug = debug

        if self.debug:
            print(spec.to_xml())

        # Change simulations iterations if you want (the policy will work best with original solver settings policy was trained on)
        # self.model.opt.ls_iterations = 50
        # self.model.opt.iterations = 50

        # For this environment we trained useing a timestep of 0.004 and frames skipped was 5
        training_timestep = 0.004
        training_frame_skip = 5
        training_action_speed = training_timestep * training_frame_skip

        # Ideally we want all inference environments to run at a timestep of 0.002 seconds per step (to help with simulation accuracy, this becomes more important the more agents you have running at the same time)
        desired_timestep = 0.002
        desired_action_speed = training_action_speed

        # Since we have a desired timestep 0.002 and we must use the same action speed, we can calculate the ideal number of frames skipped as
        desired_frame_skip = desired_action_speed / desired_timestep

        self.model.opt.timestep = desired_timestep
        self.frame_skip = int(desired_frame_skip)

        # Grab the brain neurons for backend to use
        self.path_to_brain_neurons = "agent_brains/fetch_humanoid/model"
        self.brain_structure = tuple([512, 256, 128])
        self.observation_dim = 288
        self.action_dim = 17

        # Environment constants
        self.number_of_agents = number_of_agents
        self._reset_noise_scale = 0.01
        self._agent_healthy_height = 0.3
        self._arena_radius = self.model.geom("wall_nx").size[1] - 2
        # self._ball_to_target_threshold = self.model.site("target_threshold").size[0] - self.model.geom("1_ball").size[0]
        self._ball_to_target_threshold = self.model.site("target_threshold").size[0]

        # Sites
        agent_site_idxs = [model.site(f"{i}_humanoid").id for i in range(1, number_of_agents + 1)]
        ball_site_idxs = [model.site(f"{i}_ball").id for i in range(1, number_of_agents + 1)]

        self._agent_site_idxs = jp.array(agent_site_idxs)
        self._ball_site_idxs = jp.array(ball_site_idxs)
        self._target_site_index = self.model.site("target").id

        agent_qpos_idxs = [[] for _ in range(number_of_agents)]
        agent_qvel_idxs = [[] for _ in range(number_of_agents)]
        agent_body_idxs = [[] for _ in range(number_of_agents)]
        agent_qfrc_actuator_idxs = [[] for _ in range(number_of_agents)]
        ball_qpos_idxs = [[] for _ in range(number_of_agents)]
        ball_qvel_idxs = [[] for _ in range(number_of_agents)]

        curr_qpos_idx = 0
        curr_qvel_idx = 0

        # Loop through each joint in the world model
        for joint_id in range(model.njnt):
            joint_name = model.joint(joint_id).name

            # Check if joint is a root joint
            is_root = joint_name.endswith("root")
            is_agent = "humanoid" in joint_name
            is_ball = "ball" in joint_name

            if is_root:
                if is_agent:
                    get_id = re.match(r"^(\d+)", joint_name)
                    agent_id = int(get_id.group(1)) - 1
                    agent_qpos_idxs[agent_id].extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    agent_qvel_idxs[agent_id].extend(range(curr_qvel_idx, curr_qvel_idx + 6))
                elif is_ball:
                    # Root joint of the ball
                    get_id = re.match(r"^(\d+)", joint_name)
                    agent_id = int(get_id.group(1)) - 1
                    ball_qpos_idxs[agent_id].extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    ball_qvel_idxs[agent_id].extend(range(curr_qvel_idx, curr_qvel_idx + 6))

                # Increment the index counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_idx += 7
                curr_qvel_idx += 6

            # Include only joints starting with the agent prefix
            else:
                if is_agent:
                    get_id = re.match(r"^(\d+)", joint_name)
                    agent_id = int(get_id.group(1)) - 1
                    agent_qpos_idxs[agent_id].append(curr_qpos_idx)
                    agent_qvel_idxs[agent_id].append(curr_qvel_idx)
                curr_qpos_idx += 1
                curr_qvel_idx += 1

        # Loop through each body in the world model
        for body_id in range(model.nbody):
            body_name = model.body(body_id).name
            is_agent = "humanoid" in body_name

            # Store the body indices of each agent (for Observation Space)
            if is_agent:
                get_id = re.match(r"^(\d+)", body_name)
                agent_id = int(get_id.group(1)) - 1
                agent_body_idxs[agent_id].append(body_id)

        # Loop through each DOF in the world model and store the ones that belong to the agent humanoid
        for dof_id in range(model.nv):
            for i, individual_agent_qvel_idx in enumerate(agent_qvel_idxs):
                if dof_id in individual_agent_qvel_idx:
                    agent_qfrc_actuator_idxs[i].append(dof_id)
                    break

        # Pre compute the indices of all joints before training
        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_body_idxs = jp.array(agent_body_idxs)
        self._agent_qfrc_actuator_idxs = jp.array(agent_qfrc_actuator_idxs)
        self._ball_qpos_idxs = jp.array(ball_qpos_idxs)
        self._ball_qvel_idxs = jp.array(ball_qvel_idxs)

        # Initial qpos for each agent
        start_poses = model.qpos0
        agent_start_poses = [start_poses[self._agent_qpos_idxs[i]] for i in range(number_of_agents)]
        self._agent_start_poses = jp.array(agent_start_poses)

        self._reset_ball = [True] * number_of_agents
        self._reset_agent = [True] * number_of_agents

        self._ticks = jp.zeros((number_of_agents,))

    # Resets the environment to an initial state.
    def _reset(self, data: mujoco.MjData, agent_to_reset: int, rng: jax.Array, reset_all: bool) -> Tuple[mujoco.MjData, jax.Array]:
        # Split the rng's
        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 9)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        padding = 3
        spawn_radius = self._arena_radius - padding

        # Generate random X, Y, and orientation values for the agent
        agent_pos_xy = jax.random.uniform(rng1, (2,), minval=-spawn_radius, maxval=spawn_radius)
        agent_pos_z = self._agent_start_poses[agent_to_reset][2]
        agent_pos = jp.array([agent_pos_xy[0], agent_pos_xy[1], agent_pos_z])

        azimuth = jax.random.uniform(rng2, (1,), minval=0, maxval=2 * jp.pi)[0]
        agent_orientation = jp.array([jp.cos(azimuth / 2), 0, 0, jp.sin(azimuth / 2)])

        # Get the original joint positions
        agent_joint_pos = self._agent_start_poses[agent_to_reset][7:]

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
        agent_qvel_noise = jax.random.uniform(rng5, self._agent_qvel_idxs[agent_to_reset].shape, minval=low, maxval=hi)
        ball_qpos_noise = jax.random.uniform(rng6, ball_qpos.shape, minval=low, maxval=hi)
        ball_qvel_noise = jax.random.uniform(rng7, self._ball_qvel_idxs[agent_to_reset].shape, minval=low,maxval=hi)

        # Add noise to everything
        agent_qpos_with_noise = agent_qpos + agent_qpos_noise
        ball_qpos_with_noise = ball_qpos + ball_qpos_noise
        ball_qvel_with_noise = ball_qvel + ball_qvel_noise

        if self._reset_agent[agent_to_reset] or reset_all:
            data.qpos[self._agent_qpos_idxs[agent_to_reset]] = agent_qpos_with_noise
            data.qvel[self._agent_qvel_idxs[agent_to_reset]] =  agent_qvel_noise
            data.qfrc_actuator[self._agent_qfrc_actuator_idxs[agent_to_reset]] = 0.0
            data.qacc_warmstart[self._agent_qvel_idxs[agent_to_reset]] = 0.0

        if self._reset_ball[agent_to_reset] or reset_all:
            data.qpos[self._ball_qpos_idxs[agent_to_reset]] = ball_qpos_with_noise
            data.qvel[self._ball_qvel_idxs[agent_to_reset]] = ball_qvel_with_noise
            # data.qacc_warmstart[self._ball_qvel_idxs[agent_to_reset]] = 0.0

        return data, rng

    # Resets a list of agents
    def reset(self, data: mujoco.MjData, agents_to_reset: List, rng: jax.Array, reset_all=False) -> Tuple[mujoco.MjData, jax.Array]:
        if self.debug: print("resetting")

        for agent_id, reset_bool in enumerate(agents_to_reset):
            if reset_bool:
                data, rng = self._reset(data, agent_id, rng, reset_all)
        return data, rng

    def preprocess_actions(self, actions) -> jp.ndarray:
        return actions

    def check_for_termination(self, data: mjx.Data):
        def check_for_fall(ticks, zs):
            cond = zs < 0.3
            new_ticks = jp.where(cond, (ticks + 1) % 100, ticks)
            done = cond & (new_ticks == 0)
            return new_ticks, done

        # Detect flipping by checking the z-component of the torso's orientation
        agent_height_values = data.qpos[self._agent_qpos_idxs][..., 2]
        new_ticks, fallen = check_for_fall(self._ticks, agent_height_values)
        self._ticks = new_ticks

        agent_root_pos = data.site_xpos[self._agent_site_idxs]
        global_ball_pos = data.site_xpos[self._ball_site_idxs]

        ball_out_bounds = jp.logical_or(
            jp.abs(global_ball_pos[:, 0]) > self._arena_radius + 2,
            jp.abs(global_ball_pos[:, 1]) > self._arena_radius + 2,
        )

        agent_out_bounds = jp.logical_or(
            jp.abs(agent_root_pos[:, 0]) > self._arena_radius + 2,
            jp.abs(agent_root_pos[:, 1]) > self._arena_radius + 2,
        )

        ball_reached_target = (
            jp.linalg.norm(global_ball_pos[:, :2], axis=1)
            < self._ball_to_target_threshold
        )

        # Reset when either; Ball is out of bounds, agent is out of bounds or the agent is about to flip over
        agent_reset_condition = jp.logical_or(agent_out_bounds, fallen)
        ball_reset_condition = jp.logical_or(ball_reached_target, ball_out_bounds)
        reset_condition = jp.logical_or(agent_reset_condition, ball_reset_condition)

        self._reset_ball = ball_reset_condition
        self._reset_agent = agent_reset_condition
        return reset_condition

    def _get_obs(self, data: mjx.Data, agent_id) -> jp.ndarray:
        # Get agent rotation matrix (from world to local agent frame)
        agent_frame = data.site_xmat[self._agent_site_idxs[agent_id]]

        agent_qpos = data.qpos[self._agent_qpos_idxs[agent_id]][2:]
        agent_qvel = data.qvel[self._agent_qvel_idxs[agent_id]]
        agent_cinert = data.cinert[self._agent_body_idxs[agent_id]].ravel()
        agent_cvel = data.cvel[self._agent_body_idxs[agent_id]].ravel()
        agent_qfrc_actuator = data.qfrc_actuator[self._agent_qfrc_actuator_idxs[agent_id]]

        # Get ball position relative to agent in world frame, then convert to agent frame
        ball_pos_world = data.site_xpos[self._ball_site_idxs[agent_id]] - data.site_xpos[self._agent_site_idxs[agent_id]]
        ball_pos_local = agent_frame.T @ ball_pos_world

        # Get ball velocities relative to agent in world frame, then convert to agent frame
        ball_vel_world = data.qvel[self._ball_qvel_idxs[agent_id]][:3] - data.qvel[self._agent_qvel_idxs[agent_id]][:3]
        ball_vel_local = agent_frame.T @ ball_vel_world

        ball_rotational_vel_world = data.qvel[self._ball_qvel_idxs[agent_id]][3:]
        ball_rotational_vel_local = agent_frame.T @ ball_rotational_vel_world

        # Get target position relative to agent in world frame
        target_pos_world = data.site_xpos[self._target_site_index] - data.site_xpos[self._agent_site_idxs[agent_id]]

        # Transform target position to local agent frame
        target_pos_local = agent_frame.T @ target_pos_world

        return jp.concatenate(
            [
                agent_qpos,                 # 22
                agent_qvel,                 # 23
                agent_cinert,               # 130
                agent_cvel,                 # 78
                agent_qfrc_actuator,        # 23
                ball_pos_local,             # 3
                ball_vel_local,             # 3
                ball_rotational_vel_local,  # 3
                target_pos_local,           # 3
            ]
        )

    # Vmap the observation function for each agent (since the brain is taking in JAX variables this function is can be JIT which makes this vmap method even faster)
    def get_all_obs(self, data: mjx.Data) -> jp.ndarray:
        return jax.vmap(self._get_obs, in_axes=(None, 0))(
            data, jp.arange(self.number_of_agents)
        )
