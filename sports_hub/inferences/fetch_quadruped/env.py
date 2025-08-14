import re
from typing import Any, Tuple, List
import mujoco
import jax
from jax import numpy as jp
from mujoco import mjx
import random

# Environment to help the inference engine construct observation and evaluate the state for a control policy
class FetchQuadruped_Inference:
    def __init__(self, number_of_agents=1, debug=False):
        scene_path = "sports_hub/inferences/fetch_quadruped/base_model.xml"

        # Original spec to copy from
        original_spec = mujoco.MjSpec.from_file(scene_path)

        # Main Spec that we are buliding on
        spec = mujoco.MjSpec.from_file(scene_path)

        # Remove the original quadruped body form the base model
        spec.delete(spec.body("quadruped"))

        # Create a dummy frame in the main spec and use it to attach each new quadruped too
        dummy_frame = spec.worldbody.add_frame()

        # Pre compute colours for each quadruped
        colours = []

        for i in range(number_of_agents):
            r = random.random()
            g = random.random()
            b = random.random()
            colours.append([r, g, b, 1.0])

        init_keyframe = []

        for i in range(number_of_agents):
            # Get a deep copy of quadruped body
            spec_copy = original_spec.copy()
            quadruped_body_copy = spec_copy.body("quadruped")

            # Attach it to the dummy frame of the main Spec (This is the correct way to attach bodies to specs)
            dummy_frame.attach_body(quadruped_body_copy, f"{i+1}_", "")

            # Sample a material colour
            quadruped_material = spec.material(f"{i+1}_body")
            quadruped_material.rgba = colours[i]

            # Root + quat
            init_keyframe += [0, 0, 0.55, 1.0, 0.0, 0.0, 0.0]
            # Leg joints * 4
            init_keyframe += [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]

        # Add the balls
        for i in range(number_of_agents):
            # Create a body for the ball
            ball = spec.worldbody.add_body()
            ball.name = f"{i+1}_ball"
            ball.pos = [(i + 1) * 2, (i + 1) * 2, 3]

            # Add a free joint to it, so it can roll around
            ball.add_freejoint(name=f"{i+1}_ball_root")

            #  Sample a material colour
            ball_material = spec.add_material(name=f"{i+1}_ball", rgba=colours[i])
            texture_index = mujoco.mjtTextureRole.mjTEXROLE_RGB
            ball_material.textures[texture_index] = "football"

            # Create the ball geom
            ball_geom = ball.add_geom(
                name=f"{i+1}_ball",
                density=5,
                size=[0.15, 0, 0],
                material=f"{i+1}_ball",
            )

            # Create the ball site
            ball.add_site(name=f"{i+1}_ball", size=[0.01, 0, 0], pos=[0, 0, 0.05])

            # Contacts time ...
            spec.add_pair(geomname1=f"{i+1}_quadruped_front_left_foot" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])
            spec.add_pair(geomname1=f"{i+1}_quadruped_front_right_foot" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])
            spec.add_pair(geomname1=f"{i+1}_quadruped_back_left_foot" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])
            spec.add_pair(geomname1=f"{i+1}_quadruped_back_right_foot" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])
            spec.add_pair(geomname1=f"{i+1}_quadruped_torso" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])

            spec.add_pair(geomname1="floor" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])

            spec.add_pair(geomname1="wall_nx" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])
            spec.add_pair(geomname1="wall_ny" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])
            spec.add_pair(geomname1="wall_px" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])
            spec.add_pair(geomname1="wall_py" ,geomname2=ball_geom.name, friction=[0.7, 0.7, 0.005, 0.005, 0.005], condim=6, solref=[-10000, -30])

            # Root + quat
            init_keyframe += [(i + 1) * 2, (i + 1) * 2, 0.15, 1.0, 0.0, 0.0, 0.0]

        spec.add_key(name="init",qpos=init_keyframe)

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
        self.path_to_brain_neurons = "agent_brains/fetch_quadruped/model"
        self.brain_structure = tuple([512, 256, 128, 128])
        self.observation_dim = 117
        self.action_dim = 8

        # Environment constants
        self.number_of_quadrupeds = number_of_agents
        self._reset_noise_scale = 0.1
        self._agent_healthy_upright = 0.4
        self._arena_size = self.model.geom("wall_px").size[1] - 2  # 15
        self._ball_to_target_threshold = self.model.site("target_threshold").size[0]

        # Sites
        agent_site_idxs = [model.site(f"{i}_quadruped").id for i in range(1,number_of_agents+1)]
        ball_site_idxs = [model.site(f"{i}_ball").id for i in range(1,number_of_agents+1)]

        self._agent_site_idxs = jp.array(agent_site_idxs)
        self._ball_site_idxs = jp.array(ball_site_idxs)
        self._target_site_index = self.model.site("target").id

        agent_qpos_idxs = [[] for _ in range(self.number_of_quadrupeds)]
        agent_qvel_idxs = [[] for _ in range(self.number_of_quadrupeds)]
        agent_body_idxs = [[] for _ in range(self.number_of_quadrupeds)]
        agent_qfrc_actuator_idxs = [[] for _ in range(self.number_of_quadrupeds)]
        ball_qpos_idxs = [[] for _ in range(self.number_of_quadrupeds)]
        ball_qvel_idxs = [[] for _ in range(self.number_of_quadrupeds)]

        curr_qpos_idx = 0
        curr_qvel_idx = 0

        # Loop through each joint in the world model
        for joint_id in range(model.njnt):
            joint_name = model.joint(joint_id).name

            # Check if joint is a root joint
            is_root = joint_name.endswith("root")
            is_agent = "quadruped" in joint_name
            is_ball = "ball" in joint_name

            if is_root:
                if is_agent:
                    get_id = re.match(r'^(\d+)', joint_name)
                    agent_id = int(get_id.group(1)) -1
                    # Root joint of the current agent
                    agent_qpos_idxs[agent_id].extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    agent_qvel_idxs[agent_id].extend(range(curr_qvel_idx, curr_qvel_idx + 6))
                elif is_ball:
                    # Root joint of the ball
                    get_id = re.match(r'^(\d+)', joint_name)
                    agent_id = int(get_id.group(1)) -1
                    ball_qpos_idxs[agent_id].extend(range(curr_qpos_idx, curr_qpos_idx + 7))
                    ball_qvel_idxs[agent_id].extend(range(curr_qvel_idx, curr_qvel_idx + 6))

                # Increment the index counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_idx += 7
                curr_qvel_idx += 6

            # Include only joints starting with the agent prefix
            else:
                if is_agent:
                    get_id = re.match(r'^(\d+)', joint_name)
                    agent_id = int(get_id.group(1)) -1
                    agent_qpos_idxs[agent_id].append(curr_qpos_idx)
                    agent_qvel_idxs[agent_id].append(curr_qvel_idx)
                curr_qpos_idx += 1
                curr_qvel_idx += 1

        # Loop through each body in the world model
        for body_id in range(model.nbody):
            body_name = model.body(body_id).name
            is_agent = "quadruped" in body_name

            # Store the body indices of each agent (for Observation Space)
            if is_agent:
                get_id = re.match(r'^(\d+)', body_name)
                agent_id = int(get_id.group(1)) -1
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
        start_poses = model.keyframe("init").qpos
        agent_qpos0_values = [start_poses[self._agent_qpos_idxs[i]] for i in range(number_of_agents)]
        self._agent_qpos0_values = jp.array(agent_qpos0_values)

        self._reset_ball = [True] * self.number_of_quadrupeds
        self._reset_agent = [True] * self.number_of_quadrupeds

        self._ticks = jp.zeros((self.number_of_quadrupeds,))

    # Resets the environment to an initial state.
    def _reset(self,data: mujoco.MjData, agent_to_reset:int, rng: jax.Array,reset_all:bool) -> Tuple[mujoco.MjData, jax.Array]:
        # Split the rng's
        rng, rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(rng, 7)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        padding = 2
        spawn_radius = self._arena_size - padding

        # Generate random X, Y, and orientation values for the agent
        agent_pos_xy = jax.random.uniform(rng1, (2,), minval=-spawn_radius, maxval=spawn_radius)
        agent_pos_z = self._agent_qpos0_values[agent_to_reset][2]
        agent_pos = jp.array([agent_pos_xy[0], agent_pos_xy[1], agent_pos_z])

        azimuth = jax.random.uniform(rng2, (1,), minval=0, maxval=2 * jp.pi)[0]
        agent_orientation = jp.array([jp.cos(azimuth / 2), 0, 0, jp.sin(azimuth / 2)])

        # Grab specific joint positions for the agent
        agent_qpos_positions = self._agent_qpos0_values[agent_to_reset][7:]

        # Generate joint position noise for agent
        agent_qpos_noise = jax.random.uniform(rng3, agent_qpos_positions.shape, minval=low, maxval=hi)

        # Generaten joint velocity noise for agent
        agent_qvel_noise = jax.random.uniform(rng4, self._agent_qvel_idxs[agent_to_reset][6:].shape, minval=low, maxval=hi)

        # Generate random X and Y position and velocity values for the ball
        ball_pos_xy = jax.random.uniform(rng5, (2,), minval=-spawn_radius, maxval=spawn_radius)
        ball_pos_z = 0.15
        ball_pos = jp.array([ball_pos_xy[0], ball_pos_xy[1], ball_pos_z])

        ball_orientation = jp.array([1, 0, 0, 0])

        ball_vel = jax.random.uniform(rng6, (6,), minval=-1, maxval=1)

        new_agent_qpos = jp.concatenate([agent_pos, agent_orientation, agent_qpos_positions + agent_qpos_noise])
        new_agent_qvel = jp.concatenate([jp.zeros(6), agent_qvel_noise])

        new_ball_qpos = jp.concatenate([ball_pos, ball_orientation])
        new_ball_qvel = jp.concatenate([ball_vel])

        if self._reset_agent[agent_to_reset] or reset_all:
            data.qpos[self._agent_qpos_idxs[agent_to_reset]] = new_agent_qpos
            data.qvel[self._agent_qvel_idxs[agent_to_reset]] = new_agent_qvel

        if self._reset_ball[agent_to_reset] or reset_all:
            data.qpos[self._ball_qpos_idxs[agent_to_reset]] = new_ball_qpos
            data.qvel[self._ball_qvel_idxs[agent_to_reset]] = new_ball_qvel

        return data, rng

    def reset(self, data: mujoco.MjData, agents_to_reset: List, rng: jax.Array, reset_all=False):
        if self.debug: print("resetting")
        for agent_id, do_it in enumerate(agents_to_reset):
            if do_it:
                data, rng = self._reset(data, agent_id, rng, reset_all)
        return data, rng

    def preprocess_actions(self, actions) -> jp.ndarray:
        return actions

    def check_for_termination(self, data: mjx.Data):
        def check_for_fall(ticks, orientations):
            cond = orientations < self._agent_healthy_upright
            new_ticks = jp.where(cond, (ticks + 1) % 100, ticks)
            done = cond & (new_ticks == 0)
            return new_ticks, done

        # Detect flipping by checking the z-component of the torso's orientation
        agent_orientations = jp.ravel(data.site_xmat[self._agent_site_idxs]).reshape(-1, 9)[:, 8]
        new_ticks, fallen = check_for_fall(self._ticks, agent_orientations)
        self._ticks = new_ticks

        agent_root_pos = data.site_xpos[self._agent_site_idxs]
        global_ball_pos = data.site_xpos[self._ball_site_idxs]

        ball_out_bounds = jp.logical_or(
            jp.abs(global_ball_pos[:, 0]) > self._arena_size + 2,
            jp.abs(global_ball_pos[:, 1]) > self._arena_size + 2,
        )

        agent_out_bounds = jp.logical_or(
            jp.abs(agent_root_pos[:, 0]) > self._arena_size + 2,
            jp.abs(agent_root_pos[:, 1]) > self._arena_size + 2,
        )

        ball_reached_target = jp.linalg.norm(global_ball_pos[:, :2], axis=1) < self._ball_to_target_threshold

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

        # Get agent joint positions and velocities excluding the global x and y positions
        agent_qpos = data.qpos[self._agent_qpos_idxs[agent_id]]
        agent_qvel = data.qvel[self._agent_qvel_idxs[agent_id]]


        agent_cfrc_ext = data.cfrc_ext[self._agent_body_idxs[agent_id]].ravel().clip(min=-1.0, max=1.0)

        # Get ball position relative to agent in world frame, then convert to agent frame
        ball_pos_world = (
            data.site_xpos[self._ball_site_idxs[agent_id]]
            - data.site_xpos[self._agent_site_idxs[agent_id]]
        )
        ball_pos_local = agent_frame.T @ ball_pos_world

        # Get ball velocities relative to agent in world frame, then convert to agent frame
        ball_vel_world = (
            data.qvel[self._ball_qvel_idxs[agent_id]][:3]
            - data.qvel[self._agent_qvel_idxs[agent_id]][:3]
        )
        ball_vel_local = agent_frame.T @ ball_vel_world

        ball_rotational_vel_world = data.qvel[self._ball_qvel_idxs[agent_id]][3:]
        ball_rotational_vel_local = agent_frame.T @ ball_rotational_vel_world

        # Get target position relative to agent in world frame
        target_pos_world = (
            data.site_xpos[self._target_site_index]
            - data.site_xpos[self._agent_site_idxs[agent_id]]
        )

        # Transform target position to local agent frame
        target_pos_local = agent_frame.T @ target_pos_world

        return jp.concatenate(
            [
                agent_qpos[2:],             # 13
                agent_qvel,                 # 14
                agent_cfrc_ext,             # 78
                ball_pos_local,             # 3
                ball_vel_local,             # 3
                ball_rotational_vel_local,  # 3
                target_pos_local,           # 3
            ]
        )

    # Vmap the observation function for each agent (since the brain is taking in JAX variables this function is can be JIT which makes this vmap method even faster)
    def get_all_obs(self, data: mjx.Data) -> jp.ndarray:
        return jax.vmap(self._get_obs, in_axes=(None, 0))(data, jp.arange(self.number_of_quadrupeds))

